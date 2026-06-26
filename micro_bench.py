"""Microbenchmark different rendering strategies."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import cv2
import time

import main
from main import (
    WIDTH, HEIGHT, TRIANGLE_COUNT, X1, Y1, X2, Y2, X3, Y3, FILL, ALPHA,
    TARGET_ARRAY, random_genome, render_to_array, compute_fitness,
    triangle_areas, random_triangle_rows,
)


def make_random_population(n):
    return [main.Individual() for _ in range(n)]


def bench(label, fn, repeats=3):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    best = min(times)
    print(f"{label:40s} {best*1000:8.2f} ms  ({1/best:.0f} ops/s)")


def variant_no_overlay(genome):
    """Same as current but skip overlay array entirely."""
    img = np.full((HEIGHT, WIDTH), 255, dtype=np.float32)
    points = genome[:, :FILL].reshape(TRIANGLE_COUNT, 3, 2)
    fills = genome[:, FILL]
    alphas = genome[:, ALPHA]
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    for pts, fill, alpha in zip(points, fills, alphas):
        if alpha == 0:
            continue
        if alpha == 255:
            cv2.fillConvexPoly(img, pts, float(fill))
            continue
        mask.fill(0)
        cv2.fillConvexPoly(mask, pts, 255)
        bool_mask = mask > 0
        a = alpha / 255.0
        img[bool_mask] = img[bool_mask] * (1.0 - a) + fill * a
    return img.astype(np.uint8)


def variant_no_overlay_f32(genome):
    """Same but keep float32 to avoid float64 promotion."""
    img = np.full((HEIGHT, WIDTH), 255, dtype=np.float32)
    points = genome[:, :FILL].reshape(TRIANGLE_COUNT, 3, 2)
    fills = genome[:, FILL]
    alphas = genome[:, ALPHA]
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

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


def variant_precomputed_arrays(genome, img_buf, mask_buf):
    """Reuse pre-allocated buffers (simulating per-worker buffers)."""
    img = img_buf
    img.fill(255.0)
    points = genome[:, :FILL].reshape(TRIANGLE_COUNT, 3, 2)
    fills = genome[:, FILL]
    alphas = genome[:, ALPHA]
    mask = mask_buf

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


def variant_precomputed_arrays_no_overlay_f32(genome, img_buf, mask_buf):
    img = img_buf
    img.fill(255.0)
    points = genome[:, :FILL].reshape(TRIANGLE_COUNT, 3, 2)
    fills = genome[:, FILL]
    alphas = genome[:, ALPHA]
    mask = mask_buf

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


def variant_uint8_img(genome):
    """Render using uint8 img + cv2.addWeighted for blending."""
    img = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
    points = genome[:, :FILL].reshape(TRIANGLE_COUNT, 3, 2)
    fills = genome[:, FILL]
    alphas = genome[:, ALPHA]
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    for pts, fill, alpha in zip(points, fills, alphas):
        if alpha == 0:
            continue
        if alpha == 255:
            cv2.fillConvexPoly(img, pts, int(fill))
            continue
        mask.fill(0)
        cv2.fillConvexPoly(mask, pts, 255)
        bool_mask = mask > 0
        # Use float32 img for blending precision
        # Save img[~mask]
        not_mask = ~bool_mask
        saved = img[not_mask].copy()
        # Create overlay = fill everywhere
        overlay = np.full((HEIGHT, WIDTH), int(fill), dtype=np.float32)
        # Blend
        img_f = img.astype(np.float32)
        a = alpha / 255.0
        cv2.addWeighted(img_f, 1.0 - a, overlay, a, 0, dst=img_f)
        img[bool_mask] = img_f[bool_mask].astype(np.uint8)
        img[not_mask] = saved
    return img


def main():
    # Generate a stable set of genomes to test
    rng = np.random.default_rng(42)
    genomes = [random_genome() for _ in range(20)]

    img_buf = np.full((HEIGHT, WIDTH), 255, dtype=np.float32)
    mask_buf = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    def run_current():
        for g in genomes:
            render_to_array(g)

    def run_no_overlay():
        for g in genomes:
            variant_no_overlay(g)

    def run_no_overlay_f32():
        for g in genomes:
            variant_no_overlay_f32(g)

    def run_precomputed():
        for g in genomes:
            variant_precomputed_arrays(g, img_buf, mask_buf)

    def run_precomputed_no_overlay_f32():
        for g in genomes:
            variant_precomputed_arrays_no_overlay_f32(g, img_buf, mask_buf)

    def run_uint8():
        for g in genomes:
            variant_uint8_img(g)

    # Verify correctness vs current
    for i, g in enumerate(genomes[:3]):
        a = render_to_array(g)
        b = variant_no_overlay(g)
        if not np.array_equal(a, b):
            diff = (a.astype(int) - b.astype(int))
            print(f"  genome {i}: max diff = {np.abs(diff).max()}, mean diff = {np.abs(diff).mean():.2f}")
        else:
            print(f"  genome {i}: IDENTICAL")

    print("\n--- 20 renders each ---")
    bench("Current (with overlay array)", run_current)
    bench("No overlay array", run_no_overlay)
    bench("No overlay, float32 blend", run_no_overlay_f32)
    bench("Pre-allocated buffers, no overlay", run_precomputed_no_overlay_f32)

    # Per-render time
    print("\n--- Per render ---")
    N = 200
    start = time.perf_counter()
    for _ in range(N):
        for g in genomes[:10]:
            render_to_array(g)
    elapsed = time.perf_counter() - start
    print(f"Current:   {elapsed/(N*10)*1000:.3f} ms/render")

    start = time.perf_counter()
    for _ in range(N):
        for g in genomes[:10]:
            variant_no_overlay_f32(g)
    elapsed = time.perf_counter() - start
    print(f"No overlay, f32:  {elapsed/(N*10)*1000:.3f} ms/render")

    start = time.perf_counter()
    for _ in range(N):
        for g in genomes[:10]:
            variant_precomputed_arrays_no_overlay_f32(g, img_buf, mask_buf)
    elapsed = time.perf_counter() - start
    print(f"Pre-alloc, no overlay, f32:  {elapsed/(N*10)*1000:.3f} ms/render")


if __name__ == "__main__":
    main()