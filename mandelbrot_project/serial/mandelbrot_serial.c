/**
 * @file mandelbrot_serial.c
 * @brief High-Performance Serial Reference Implementation of the Mandelbrot Set
 * @course SE3082 - Parallel Computing (Assignment 3)
 * 
 * DESIGN PRINCIPLES:
 * 1. Cache-Friendly Memory Layout:
 *    Instead of a nested pointer structure (e.g., int**), we allocate a contiguous 1D
 *    memory buffer of size (width * height). This ensures spatial locality during
 *    row-major traversal, allowing the CPU cache prefetcher to fetch subsequent pixels 
 *    efficiently and minimizing cache line thrashing.
 * 
 * 2. Flat Indexing:
 *    Any pixel at coordinates (x, y) maps to index: [y * width + x].
 * 
 * 3. Robust High-Resolution Timing:
 *    Utilizes OS-specific high-resolution counters (QueryPerformanceCounter on Windows 
 *    and clock_gettime on POSIX platforms) to accurately measure wall-clock execution time,
 *    excluding I/O operations, for clean speedup comparisons.
 * 
 * 4. Image Output:
 *    Saves the iteration matrix as a standard binary Portable Graymap (PGM) image (format P5).
 *    Iteration values are normalized to a 0-255 grayscale range for direct binary correctness checking.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#endif

/* Default configuration parameters */
#define DEFAULT_WIDTH 1600
#define DEFAULT_HEIGHT 1200
#define DEFAULT_MAX_ITER 1000
#define DEFAULT_OUTPUT "mandelbrot_serial.pgm"

/* Define Mandelbrot complex plane coordinate bounds */
#define X_MIN -2.0
#define X_MAX 1.0
#define Y_MIN -1.5
#define Y_MAX 1.5
#define ESCAPE_RADIUS_SQ 4.0 // Standard escape threshold (r=2.0, so r^2=4.0)

/**
 * @brief Retrieves high-resolution wall-clock time in seconds.
 * Handles platform differences between POSIX systems and Windows.
 */
double get_wall_time() {
#ifdef _WIN32
    LARGE_INTEGER time, freq;
    if (!QueryPerformanceFrequency(&freq)) {
        return 0.0;
    }
    QueryPerformanceCounter(&time);
    return (double)time.QuadPart / freq.QuadPart;
#else
    struct timespec time;
    if (clock_gettime(CLOCK_MONOTONIC, &time) != 0) {
        return 0.0;
    }
    return (double)time.tv_sec + (double)time.tv_nsec * 1e-9;
#endif
}

/**
 * @brief Saves the computation buffer as a P5 binary PGM image file.
 * Normalizes iteration count to [0, 255] grayscale intensity.
 * 
 * @param filename Path to save the PGM file.
 * @param width Width of the image.
 * @param height Height of the image.
 * @param max_iter Maximum iterations limit.
 * @param image_buffer Pointer to the flat 1D iteration count array.
 */
void save_pgm(const char *filename, int width, int height, int max_iter, const int *image_buffer) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error: Could not open output file '%s' for writing.\n", filename);
        return;
    }

    /* PGM P5 header format: 
     * P5
     * <width> <height>
     * <max_gray_value>
     */
    fprintf(fp, "P5\n%d %d\n255\n", width, height);

    /* Write normalized grayscale values to file */
    for (int i = 0; i < width * height; i++) {
        /* Points inside the set (iter == max_iter) are drawn black (0 value).
         * Points outside are shaded based on how quickly they escaped.
         */
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

    printf("====================================================\n");
    printf("Mandelbrot Serial Baseline Reference Code\n");
    printf("Resolution : %d x %d pixels\n", width, height);
    printf("Max Iter   : %d\n", max_iter);
    printf("Output File: %s\n", output_file);
    printf("====================================================\n");

    /* Allocate the flat 1D buffer on the heap */
    int *image_buffer = (int *)malloc(width * height * sizeof(int));
    if (!image_buffer) {
        fprintf(stderr, "Fatal Error: Memory allocation failed for image buffer.\n");
        return EXIT_FAILURE;
    }

    printf("Computing Mandelbrot set sequentially...\n");
    double start_time = get_wall_time();

    /* 
     * Core Mandelbrot loop:
     * We traverse the domain in a row-major fashion (y then x) to match the 1D buffer layout.
     * This ensures sequential writes, maximizing CPU L1/L2 cache efficiency.
     */
    for (int y = 0; y < height; y++) {
        /* Calculate the imaginary coordinate component for the current row */
        double ci = Y_MIN + (double)y * (Y_MAX - Y_MIN) / (double)height;

        for (int x = 0; x < width; x++) {
            /* Calculate the real coordinate component for the current pixel */
            double cr = X_MIN + (double)x * (X_MAX - X_MIN) / (double)width;

            double zr = 0.0;
            double zi = 0.0;
            int iter = 0;

            /* 
             * Quadratic recurrence calculation: z_{k+1} = z_k^2 + c
             * Expanding:
             * Real part: zr_{k+1} = zr^2 - zi^2 + cr
             * Imag part: zi_{k+1} = 2 * zr * zi + ci
             */
            double zr2 = 0.0;
            double zi2 = 0.0;

            /* Optimizing by caching squared terms to avoid duplicate multiplications */
            while (zr2 + zi2 <= ESCAPE_RADIUS_SQ && iter < max_iter) {
                double temp = zr2 - zi2 + cr;
                zi = 2.0 * zr * zi + ci;
                zr = temp;
                
                zr2 = zr * zr;
                zi2 = zi * zi;
                iter++;
            }

            /* Record the iteration count at which escape occurred (or limit reached) */
            image_buffer[y * width + x] = iter;
        }
    }

    double end_time = get_wall_time();
    double elapsed_time = end_time - start_time;
    printf("Computation completed in %f seconds.\n", elapsed_time);

    /* Output baseline image for visualization and correctness check */
    save_pgm(output_file, width, height, max_iter, image_buffer);

    /* Free heap memory to prevent resource leaks */
    free(image_buffer);

    return EXIT_SUCCESS;
}
