from PIL import Image, ImageTk
from tkinter import Tk, Label
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from multiprocessing import shared_memory
import argparse
import os
import numpy as np
import cv2
import random
import time

WIDTH = 128
HEIGHT = 128

TRIANGLE_COUNT = 50
POPULATION_SIZE = 100
ELITE_SIZE = 20
SAVE_EVERY = 50
REPORT_EVERY = 25
VIEWER_EVERY = 25
MUTATIONS_PER_CALL = 2

MUTATION_STEPS = (3, 15, 40) 
REPLACE_TRIANGLE_PROB = 0.10


TOURNAMENT_SIZE = 5 
CROSSOVER_RATE = 0.85
FITNESS_WORKERS = min(7, os.cpu_count() or 1)
FITNESS_CHUNK_SIZE = 14

cv2.setUseOptimized(True)
cv2.setNumThreads(1)

X1, Y1, X2, Y2, X3, Y3, FILL, ALPHA = range(8)

TARGET = Image.open("output/target.png").convert("L")
TARGET_ARRAY = np.array(TARGET, dtype=np.uint8)
WORKER_SHARED_MEMORY = None
WORKER_GENOMES = None


class Individual:
    __slots__ = ("fitness", "genome", "points", "fills", "alphas")

    def __init__(self, genome=None, fitness=None):
        self.fitness = fitness
        self.genome = random_genome() if genome is None else genome
        self.points = self.genome[:, :FILL].reshape(TRIANGLE_COUNT, 3, 2)
        self.fills = self.genome[:, FILL]
        self.alphas = self.genome[:, ALPHA]


def triangle_areas(rows):
    return (
        rows[:, X1] * (rows[:, Y2] - rows[:, Y3])
        + rows[:, X2] * (rows[:, Y3] - rows[:, Y1])
        + rows[:, X3] * (rows[:, Y1] - rows[:, Y2])
    )


def random_triangle_rows(count):
    rows = np.empty((count, 8), dtype=np.int32)
    written = 0

    while written < count:
        batch_size = max((count - written) * 2, 8)
        batch = np.empty((batch_size, 8), dtype=np.int32)
        batch[:, X1] = np.random.randint(0, WIDTH, size=batch_size)
        batch[:, Y1] = np.random.randint(0, HEIGHT, size=batch_size)
        batch[:, X2] = np.random.randint(0, WIDTH, size=batch_size)
        batch[:, Y2] = np.random.randint(0, HEIGHT, size=batch_size)
        batch[:, X3] = np.random.randint(0, WIDTH, size=batch_size)
        batch[:, Y3] = np.random.randint(0, HEIGHT, size=batch_size)
        batch[:, FILL] = np.random.randint(0, 256, size=batch_size)
        # No bias applied here, strictly 0 to 255
        batch[:, ALPHA] = np.random.randint(0, 256, size=batch_size) 

        valid = batch[triangle_areas(batch) != 0]
        take = min(count - written, len(valid))
        rows[written:written + take] = valid[:take]
        written += take

    return rows


def random_genome():
    return random_triangle_rows(TRIANGLE_COUNT)


# Optimized Rendering with Pre-allocated Masks
def render_to_array(genome, points=None, fills=None, alphas=None, out=None):
    img = np.full((HEIGHT, WIDTH), 255, dtype=np.float32)

    points = genome[:, :FILL].reshape(TRIANGLE_COUNT, 3, 2) if points is None else points
    fills = genome[:, FILL] if fills is None else fills
    alphas = genome[:, ALPHA] if alphas is None else alphas

    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    for pts, fill, alpha in zip(points, fills, alphas):
        if alpha == 0:
            continue

        if alpha == 255:
            cv2.fillConvexPoly(img, pts, color=float(fill))
            continue

        mask.fill(0)
        cv2.fillConvexPoly(mask, pts, 255)
        bool_mask = mask > 0

        a_f = np.float32(alpha) * np.float32(1.0 / 255.0)
        inv_a_f = np.float32(1.0) - a_f
        fill_f = np.float32(fill)
        img[bool_mask] = img[bool_mask] * inv_a_f + fill_f * a_f

    if out is None:
        return img.astype(np.uint8)
    np.copyto(out, img, casting="unsafe")
    return out


def render_to_image(genome):
    return Image.fromarray(render_to_array(genome))


# Fitness

def compute_fitness(arr):
    return int(cv2.norm(arr, TARGET_ARRAY, cv2.NORM_L1))


def compute_genome_fitness(genome):
    return compute_fitness(render_to_array(genome))


def init_worker_shared_memory(name, shape):
    global WORKER_SHARED_MEMORY, WORKER_GENOMES, WORKER_IMG_BUF, WORKER_MASK_BUF
    WORKER_SHARED_MEMORY = shared_memory.SharedMemory(name=name)
    WORKER_GENOMES = np.ndarray(shape, dtype=np.int32, buffer=WORKER_SHARED_MEMORY.buf)
    WORKER_IMG_BUF = np.full((HEIGHT, WIDTH), 255, dtype=np.float32)
    WORKER_MASK_BUF = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)


def render_to_array_shared(genome):
    img = WORKER_IMG_BUF
    img.fill(255.0)
    points = genome[:, :FILL].reshape(TRIANGLE_COUNT, 3, 2)
    fills = genome[:, FILL]
    alphas = genome[:, ALPHA]
    mask = WORKER_MASK_BUF

    for pts, fill, alpha in zip(points, fills, alphas):
        if alpha == 0:
            continue
        if alpha == 255:
            cv2.fillConvexPoly(img, pts, float(fill))
            continue
        mask.fill(0)
        cv2.fillConvexPoly(mask, pts, 255)
        bool_mask = mask > 0
        a_f = np.float32(alpha) * np.float32(1.0 / 255.0)
        inv_a_f = np.float32(1.0) - a_f
        fill_f = np.float32(fill)
        img[bool_mask] = img[bool_mask] * inv_a_f + fill_f * a_f

    return img.astype(np.uint8)


def compute_shared_fitness_range(bounds):
    start, end = bounds
    return [int(cv2.norm(render_to_array_shared(WORKER_GENOMES[i]), TARGET_ARRAY, cv2.NORM_L1)) for i in range(start, end)]


def compute_individual_fitness(individual):
    return compute_fitness(render_to_array(individual.genome, individual.points, individual.fills, individual.alphas))


# GA operations

def evaluate_population(population, executor=None):
    missing = [ind for ind in population if ind.fitness is None]
    if not missing:
        return

    if executor is not None and hasattr(executor, "evaluate"):
        executor.evaluate(missing)
        return

    for ind in missing:
        ind.fitness = compute_individual_fitness(ind)


def sort_population(population):
    population.sort(key=lambda ind: ind.fitness)


def clone_individual(parent):
    return Individual(parent.genome.copy(), parent.fitness)


def tournament_indices(count):
    return np.random.randint(0, POPULATION_SIZE, size=(count, TOURNAMENT_SIZE)).min(axis=1)


def create_next_generation(population):
    next_pop = [clone_individual(e) for e in population[:ELITE_SIZE]]
    child_count = POPULATION_SIZE - ELITE_SIZE
    population_genomes = np.stack([ind.genome for ind in population])

    parent1_indices = tournament_indices(child_count)
    child_genomes = population_genomes[parent1_indices].copy()

    crossover_mask = np.random.random(child_count) < CROSSOVER_RATE
    crossover_rows = np.flatnonzero(crossover_mask)
    
    # 1-POINT CROSSOVER: Preserves foreground/background layering
    if crossover_rows.size:
        parent2_indices = tournament_indices(crossover_rows.size)
        parent2_genomes = population_genomes[parent2_indices]
        
        # Pick a random slice point in the stack of triangles
        cut_points = np.random.randint(1, TRIANGLE_COUNT, size=crossover_rows.size)
        tri_indices = np.arange(TRIANGLE_COUNT)
        
        # True for triangles before the cut, False after
        keep_parent1_mask = tri_indices[None, :] < cut_points[:, None]
        
        # Keep Parent 1's bottom layers, take Parent 2's top layers
        child_genomes[crossover_rows] = np.where(
            keep_parent1_mask[:, :, None],
            child_genomes[crossover_rows],
            parent2_genomes
        )

    mutate_child_genomes(child_genomes)
    next_pop.extend(Individual(genome) for genome in child_genomes)
    return next_pop


def mutate_child_genomes(child_genomes):
    child_count = len(child_genomes)
    child_indices = np.arange(child_count)

    for _ in range(MUTATIONS_PER_CALL):
        triangle_indices = np.random.randint(0, TRIANGLE_COUNT, size=child_count)
        replace_mask = np.random.random(child_count) < REPLACE_TRIANGLE_PROB

        replace_count = int(replace_mask.sum())
        if replace_count:
            child_genomes[
                child_indices[replace_mask],
                triangle_indices[replace_mask],
            ] = random_triangle_rows(replace_count)

        edit_mask = ~replace_mask
        edit_count = int(edit_mask.sum())
        if not edit_count:
            continue

        edit_child_indices = child_indices[edit_mask]
        edit_triangle_indices = triangle_indices[edit_mask]
        params = np.random.randint(0, 8, size=edit_count)
        
        # DECOUPLED MUTATION STEPS
        # Small, precise steps for Fill and Alpha
        steps = np.random.choice([2, 5, 10], size=edit_count)
        
        # Big, exploratory steps for X and Y coordinates
        coord_mask = (params == X1) | (params == X2) | (params == X3) | \
                     (params == Y1) | (params == Y2) | (params == Y3)
                     
        if coord_mask.any():
            steps[coord_mask] = np.random.choice([5, 15, 40], size=coord_mask.sum())

        deltas = np.random.randint(-steps, steps + 1)

        child_genomes[edit_child_indices, edit_triangle_indices, params] += deltas

        fill_mask = params == FILL
        alpha_mask = params == ALPHA 
        x_mask = (params == X1) | (params == X2) | (params == X3)
        y_mask = ~(fill_mask | x_mask | alpha_mask) 

        if fill_mask.any():
            idx = edit_child_indices[fill_mask]
            tri = edit_triangle_indices[fill_mask]
            child_genomes[idx, tri, FILL] = np.clip(child_genomes[idx, tri, FILL], 0, 255)

        if alpha_mask.any():
            idx = edit_child_indices[alpha_mask]
            tri = edit_triangle_indices[alpha_mask]
            child_genomes[idx, tri, ALPHA] = np.clip(child_genomes[idx, tri, ALPHA], 0, 255)

        if x_mask.any():
            idx = edit_child_indices[x_mask]
            tri = edit_triangle_indices[x_mask]
            param = params[x_mask]
            child_genomes[idx, tri, param] = np.clip(child_genomes[idx, tri, param], 0, WIDTH - 1)

        if y_mask.any():
            idx = edit_child_indices[y_mask]
            tri = edit_triangle_indices[y_mask]
            param = params[y_mask]
            child_genomes[idx, tri, param] = np.clip(child_genomes[idx, tri, param], 0, HEIGHT - 1)


# Tkinter viewer

def update_viewer(root, image_label, info_label, image, generation, fit):
    photo = ImageTk.PhotoImage(image)
    image_label.configure(image=photo)
    image_label.image = photo
    info_label.configure(text=f"Generation: {generation}\nFitness: {fit:,}")
    root.update_idletasks()
    root.update()


class SharedFitnessPool:
    def __init__(self, workers, chunk_size):
        self.workers = workers
        self.chunk_size = chunk_size
        self.shape = (POPULATION_SIZE, TRIANGLE_COUNT, 8)
        self.shared_memory = None
        self.genomes = None
        self.executor = None

    def __enter__(self):
        self.shared_memory = shared_memory.SharedMemory(
            create=True,
            size=int(np.prod(self.shape) * np.dtype(np.int32).itemsize),
        )
        self.genomes = np.ndarray(self.shape, dtype=np.int32, buffer=self.shared_memory.buf)
        self.executor = ProcessPoolExecutor(
            max_workers=self.workers,
            initializer=init_worker_shared_memory,
            initargs=(self.shared_memory.name, self.shape),
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        self.executor.shutdown()
        self.shared_memory.close()
        self.shared_memory.unlink()

    def evaluate(self, missing):
        if len(missing) < self.chunk_size * 2:
            for ind in missing:
                ind.fitness = compute_individual_fitness(ind)
            return

        self.genomes[:len(missing)] = np.stack([ind.genome for ind in missing])

        ranges = [
            (start, min(start + self.chunk_size, len(missing)))
            for start in range(0, len(missing), self.chunk_size)
        ]

        index = 0
        for chunk in self.executor.map(compute_shared_fitness_range, ranges):
            for fitness in chunk:
                missing[index].fitness = fitness
                index += 1


def make_fitness_executor(enabled=True):
    if not enabled or FITNESS_WORKERS <= 1:
        return nullcontext(None)
    return SharedFitnessPool(FITNESS_WORKERS, FITNESS_CHUNK_SIZE)


def benchmark(generations, parallel=True):
    population = [Individual() for _ in range(POPULATION_SIZE)]
    best_fitness = None
    start = time.perf_counter()

    with make_fitness_executor(parallel) as executor:
        for _ in range(generations):
            evaluate_population(population, executor)
            sort_population(population)
            best_fitness = population[0].fitness
            population = create_next_generation(population)

    elapsed = time.perf_counter() - start
    print(
        f"{generations / elapsed:.2f} gen/s, "
        f"{elapsed / generations * 1000:.2f} ms/gen, best={best_fitness}"
    )


# Main loop

def main(parallel=True):
    root = Tk()
    root.title("Genetic Programming - Pure GA with Alpha")
    info_label = Label(root, font=("Consolas", 14))
    info_label.pack()
    image_label = Label(root)
    image_label.pack()

    population = [Individual() for _ in range(POPULATION_SIZE)]
    generation = 0
    best_fitness = float("inf")
    best_genome = None
    last_report = time.perf_counter()

    with make_fitness_executor(parallel) as executor:
        while True:
            evaluate_population(population, executor)
            sort_population(population)

            best = population[0]

            if best.fitness < best_fitness:
                best_fitness = best.fitness
                best_genome = best.genome.copy()

            if generation % VIEWER_EVERY == 0 and best_genome is not None:
                update_viewer(
                    root,
                    image_label,
                    info_label,
                    render_to_image(best_genome),
                    generation,
                    best_fitness,
                )

            if generation % REPORT_EVERY == 0:
                now = time.perf_counter()
                gens_per_second = REPORT_EVERY / (now - last_report) if generation else 0
                last_report = now
                print(
                    f"Generation {generation:6d} | Fitness {best.fitness:,} | "
                    f"{gens_per_second:.1f} gen/s"
                )

            if generation % SAVE_EVERY == 0 and best_genome is not None:
                render_to_image(best_genome).save("output/best.png")

            population = create_next_generation(population)
            generation += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=int, metavar="GENERATIONS")
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--workers", type=int, default=FITNESS_WORKERS)
    parser.add_argument("--chunk-size", type=int, default=FITNESS_CHUNK_SIZE)
    args = parser.parse_args()

    FITNESS_WORKERS = max(1, args.workers)
    FITNESS_CHUNK_SIZE = max(1, args.chunk_size)
    parallel = not args.serial

    if args.benchmark:
        benchmark(args.benchmark, parallel=parallel)
    else:
        main(parallel=parallel)