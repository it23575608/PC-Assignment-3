/**
 * @file mandelbrot_openmp.c
 * @brief High-Performance Shared-Memory Parallel Mandelbrot Implementation using OpenMP
 * @course SE3082 - Parallel Computing (Assignment 3)
 * 
 * PARALLEL DESIGN ARCHITECTURE:
 * 1. Loop-Level Parallelism:
 *    We parallelize the outer loop (y-axis / rows) using OpenMP. Parallelizing the outer
 *    loop maximizes granularity (chunks of work) and keeps threads active with minimal 
 *    fork-join overhead compared to parallelizing the inner loop.
 * 
 * 2. Mitigating Inherent Load Imbalance:
 *    The Mandelbrot set represents an extremely irregular computational workload. Points
 *    inside the main cardioid/bulbs require `max_iter` iterations, while points in the
 *    outer blue/space escape after a few iterations. 
 *    - Static Scheduling (e.g., schedule(static)) would divide the rows evenly. Threads
 *      assigned central rows (heavy workload) would bottleneck the execution, while threads 
 *      assigned outer rows would finish instantly and sit idle.
 *    - Dynamic Scheduling (schedule(dynamic, 16)) distributes rows dynamically in blocks of 16 
 *      to threads as they become available. This forms a "work queue" mechanism that balances 
 *      the load across cores.
 * 
 * 3. Race-Condition Prevention (Data Scoping):
 *    To ensure thread safety, all variables representing coordinates and intermediate 
 *    values (x, cr, ci, zr, zi, zr2, zi2, iter) must be thread-private. We declare these 
 *    variables inside the parallel loop scope. In C99, variables declared inside the body 
 *    of an OpenMP loop are implicitly private to each thread, eliminating data races.
 * 
 * 4. Microsecond Timing:
 *    Utilizes `omp_get_wtime()` to record high-resolution wall-clock duration of the parallel execution.
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/* Default configuration parameters */
#define DEFAULT_WIDTH 1600
#define DEFAULT_HEIGHT 1200
#define DEFAULT_MAX_ITER 1000
#define DEFAULT_OUTPUT "mandelbrot_openmp.pgm"

/* Define Mandelbrot complex plane coordinate bounds */
#define X_MIN -2.0
#define X_MAX 1.0
#define Y_MIN -1.5
#define Y_MAX 1.5
#define ESCAPE_RADIUS_SQ 4.0 // Escape threshold (r^2 = 4.0)

/**
 * @brief Saves the computation buffer as a P5 binary PGM image file.
 * Must match the serial baseline serialization exactly for binary verification.
 */
void save_pgm(const char *filename, int width, int height, int max_iter, const int *image_buffer) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error: Could not open output file '%s' for writing.\n", filename);
        return;
    }

    /* PGM P5 header */
    fprintf(fp, "P5\n%d %d\n255\n", width, height);

    /* Write normalized grayscale values to file */
    for (int i = 0; i < width * height; i++) {
        unsigned char color = 0;
        if (image_buffer[i] < max_iter) {
            color = (unsigned char)((image_buffer[i] * 255) / max_iter);
        }
        fputc(color, fp);
    }

    fclose(fp);
    printf("Successfully wrote Mandelbrot image to '%s'\n", filename);
}

int main(int argc, char *argv[]) {
    int width = DEFAULT_WIDTH;
    int height = DEFAULT_HEIGHT;
    int max_iter = DEFAULT_MAX_ITER;
    const char *output_file = DEFAULT_OUTPUT;

    /* Parse command line arguments if provided */
    if (argc > 1) width = atoi(argv[1]);
    if (argc > 2) height = atoi(argv[2]);
    if (argc > 3) max_iter = atoi(argv[3]);
    if (argc > 4) output_file = argv[4];

    /* Find the maximum number of threads available/configured */
    int max_threads = omp_get_max_threads();

    printf("====================================================\n");
    printf("Mandelbrot Parallel Implementation (OpenMP)\n");
    printf("Resolution       : %d x %d pixels\n", width, height);
    printf("Max Iterations   : %d\n", max_iter);
    printf("Configured Cores : %d\n", max_threads);
    printf("Output File      : %s\n", output_file);
    printf("====================================================\n");

    /* Allocate the flat 1D buffer on the heap */
    int *image_buffer = (int *)malloc(width * height * sizeof(int));
    if (!image_buffer) {
        fprintf(stderr, "Fatal Error: Memory allocation failed for image buffer.\n");
        return EXIT_FAILURE;
    }

    printf("Computing Mandelbrot set in parallel...\n");
    double start_time = omp_get_wtime();

    /* 
     * OpenMP Parallel Work Sharing:
     * - `parallel for` spawns a team of threads and distributes loop iterations.
     * - `schedule(dynamic, 16)` assigns chunks of 16 rows to threads dynamically. 
     *   This addresses the severe load imbalance between outer and inner points.
     * - `shared(image_buffer, width, height, max_iter)` specifies the parameters and 
     *   output destination which are accessed concurrently.
     * - Scoped variables declared inside the loop (y, ci, x, cr, zr, zi, zr2, zi2, iter)
     *   are implicitly `private` to each thread.
     */
    #pragma omp parallel for schedule(dynamic, 16) shared(image_buffer, width, height, max_iter)
    for (int y = 0; y < height; y++) {
        /* Calculate imaginary coordinate component for the current row */
        double ci = Y_MIN + (double)y * (Y_MAX - Y_MIN) / (double)height;

        for (int x = 0; x < width; x++) {
            /* Calculate real coordinate component for the current pixel */
            double cr = X_MIN + (double)x * (X_MAX - X_MIN) / (double)width;

            double zr = 0.0;
            double zi = 0.0;
            double zr2 = 0.0;
            double zi2 = 0.0;
            int iter = 0;

            /* Compute quadratic recurrence z_{k+1} = z_k^2 + c */
            while (zr2 + zi2 <= ESCAPE_RADIUS_SQ && iter < max_iter) {
                double temp = zr2 - zi2 + cr;
                zi = 2.0 * zr * zi + ci;
                zr = temp;

                zr2 = zr * zr;
                zi2 = zi * zi;
                iter++;
            }

            /* Record iteration result in the flat 1D array.
             * Writes are race-free because each thread computes and writes to a unique index (y * width + x).
             */
            image_buffer[y * width + x] = iter;
        }
    }

    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;
    printf("Parallel computation completed in %f seconds.\n", elapsed_time);

    /* Output parallelized image - must be bit-for-bit identical to the serial baseline output */
    save_pgm(output_file, width, height, max_iter, image_buffer);

    /* Clean up memory resources */
    free(image_buffer);

    return EXIT_SUCCESS;
}
