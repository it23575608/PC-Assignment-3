/*
 * Sequential reference implementation of the Mandelbrot set
 * Course: SE3082 - Parallel Computing
 */

#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#endif

#define DEFAULT_WIDTH 1600
#define DEFAULT_HEIGHT 1200
#define DEFAULT_MAX_ITER 1000
#define DEFAULT_OUTPUT "mandelbrot_serial.pgm"

#define X_MIN -2.0
#define X_MAX 1.0
#define Y_MIN -1.5
#define Y_MAX 1.5
#define ESCAPE_RADIUS_SQ 4.0

// Function to retrieve high-resolution time in seconds
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

// Function to save image in PGM format
void save_pgm(const char *filename, int width, int height, int max_iter, const int *image_buffer) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error: Could not open output file '%s' for writing.\n", filename);
        return;
    }

    // Write P5 PGM header
    fprintf(fp, "P5\n%d %d\n255\n", width, height);

    // Save pixel values
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

    // Parse command line arguments if provided
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

    // Allocate memory buffer
    int *image_buffer = (int *)malloc(width * height * sizeof(int));
    if (!image_buffer) {
        fprintf(stderr, "Fatal Error: Memory allocation failed for image buffer.\n");
        return EXIT_FAILURE;
    }

    printf("Computing Mandelbrot set sequentially...\n");
    double start_time = get_wall_time();

    // Traverse the domain in row-major order
    for (int y = 0; y < height; y++) {
        double ci = Y_MIN + (double)y * (Y_MAX - Y_MIN) / (double)height;

        for (int x = 0; x < width; x++) {
            double cr = X_MIN + (double)x * (X_MAX - X_MIN) / (double)width;

            double zr = 0.0;
            double zi = 0.0;
            double zr2 = 0.0;
            double zi2 = 0.0;
            int iter = 0;

            // Mandelbrot escape loop
            while (zr2 + zi2 <= ESCAPE_RADIUS_SQ && iter < max_iter) {
                double temp = zr2 - zi2 + cr;
                zi = 2.0 * zr * zi + ci;
                zr = temp;
                
                zr2 = zr * zr;
                zi2 = zi * zi;
                iter++;
            }

            image_buffer[y * width + x] = iter;
        }
    }

    double end_time = get_wall_time();
    double elapsed_time = end_time - start_time;
    printf("Computation completed in %f seconds.\n", elapsed_time);

    // Save final PGM image
    save_pgm(output_file, width, height, max_iter, image_buffer);

    free(image_buffer);
    return EXIT_SUCCESS;
}
