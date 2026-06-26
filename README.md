# Genetic Program

A genetic algorithm that reconstructs a grayscale target image by evolving a
population of semi-transparent triangles.

Each individual in the population is a genome of 50 triangles. Every triangle
has three vertices, a grayscale fill value, and an alpha value. The algorithm
renders each genome to a 128x128 image and compares it against the target using
the L1 (mean absolute) difference. Lower is better.

## How it works

- **Genome**: an array of triangles, each defined by 3 points, a fill color,
  and an alpha.
- **Fitness**: L1 pixel distance between the rendered image and the target.
- **Selection**: tournament selection picks parents.
- **Crossover**: one-point crossover that preserves foreground/background
  layering of triangles.
- **Mutation**: small precise steps for fill/alpha, larger exploratory steps
  for coordinates. Occasionally a triangle is replaced entirely.
- **Elitism**: the top individuals from each generation are carried over
  unchanged.

Fitness evaluation is parallelized across CPU cores using shared memory and a
process pool, with pre-allocated render buffers per worker to avoid repeated
allocations.

## Project layout

```
main.py          GA implementation, rendering, and run loop
pre_process.py   Converts a source image into the grayscale 128x128 target
micro_bench.py   Microbenchmarks for different rendering strategies
input/           Source images used to build targets
output/          Generated targets and best-so-far results
```

## Setup

Requires Python 3.13+ and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

## Prepare a target

Edit the source path in `pre_process.py` if needed, then run:

```bash
uv run pre_process.py
```

This writes `output/target.png`.

## Run

Run the GA with the live Tkinter viewer:

```bash
uv run main.py
```

The viewer shows the current best render, generation count, and fitness. The
best image is periodically saved to `output/best.png`.

### Options

```
--benchmark GENERATIONS   Run headless for a number of generations and print throughput
--serial                  Disable the process pool and evaluate fitness in-process
--workers N               Number of fitness worker processes (default: up to 7)
--chunk-size N            Genomes evaluated per task chunk sent to each worker
```

Example benchmark:

```bash
uv run main.py --benchmark 500
```

## Dependencies

- numpy
- opencv-python
- pillow
- scikit-image
