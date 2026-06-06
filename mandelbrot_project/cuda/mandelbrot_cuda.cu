/*
 * Mandelbrot set generation using NVIDIA CUDA
 * Course: SE3082 - Parallel Computing
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define DEFAULT_WIDTH 1600
#define DEFAULT_HEIGHT 1200
#define DEFAULT_MAX_ITER 1000
#define DEFAULT_OUTPUT "mandelbrot_cuda.pgm"

#define X_MIN -2.0
#define X_MAX 1.0
#define Y_MIN -1.5
#define Y_MAX 1.5
#define ESCAPE_RADIUS_SQ 4.0

// CUDA Kernel for Mandelbrot set calculation
__global__ void mandelbrot_kernel(int *d_image_buffer, int width, int height, int max_iter,
                                  double x_min, double x_max, double y_min, double y_max) {
    // Map thread indices to 2D image coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check
    if (x < width && y < height) {
        double cr = x_min + (double)x * (x_max - x_min) / (double)width;
        double ci = y_min + (double)y * (y_max - y_min) / (double)height;

        double zr = 0.0;
        double zi = 0.0;
        double zr2 = 0.0;
        double zi2 = 0.0;
        int iter = 0;

        // Mandelbrot iteration loop
        while (zr2 + zi2 <= ESCAPE_RADIUS_SQ && iter < max_iter) {
            double temp = zr2 - zi2 + cr;
            zi = 2.0 * zr * zi + ci;
            zr = temp;
            
            zr2 = zr * zr;
            zi2 = zi * zi;
            iter++;
        }

        d_image_buffer[y * width + x] = iter;
    }
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
    int block_x = 16;
    int block_y = 16;

    // Parse command line arguments if provided
    if (argc > 1) width = atoi(argv[1]);
    if (argc > 2) height = atoi(argv[2]);
    if (argc > 3) max_iter = atoi(argv[3]);
    if (argc > 4) output_file = argv[4];
    if (argc > 5) block_x = atoi(argv[5]);
    if (argc > 6) block_y = atoi(argv[6]);

    printf("====================================================\n");
    printf("Mandelbrot Parallel Implementation (CUDA GPU)\n");
    printf("Resolution       : %d x %d pixels\n", width, height);
    printf("Max Iterations   : %d\n", max_iter);
    printf("Output File      : %s\n", output_file);
    printf("====================================================\n");

    // Allocate host memory buffer
    int *h_image_buffer = (int *)malloc(width * height * sizeof(int));
    if (!h_image_buffer) {
        fprintf(stderr, "Fatal Error: Host memory allocation failed.\n");
        return EXIT_FAILURE;
    }

    // Allocate device/GPU memory buffer
    int *d_image_buffer = NULL;
    cudaError_t err = cudaMalloc((void **)&d_image_buffer, width * height * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Fatal Error: Device memory allocation failed: %s\n", cudaGetErrorString(err));
        free(h_image_buffer);
        return EXIT_FAILURE;
    }

    // Define thread block size and calculate grid dimensions
    dim3 block_dim(block_x, block_y);
    dim3 grid_dim((width + block_dim.x - 1) / block_dim.x, 
                  (height + block_dim.y - 1) / block_dim.y);

    printf("Launching CUDA Kernel with block size %dx%d and grid size %dx%d...\n", 
           block_dim.x, block_dim.y, grid_dim.x, grid_dim.y);

    // Set up CUDA events for execution timing
    cudaEvent_t start_evt, stop_evt;
    cudaEventCreate(&start_evt);
    cudaEventCreate(&stop_evt);

    cudaEventRecord(start_evt, 0);

    // Launch CUDA kernel
    mandelbrot_kernel<<<grid_dim, block_dim>>>(d_image_buffer, width, height, max_iter, 
                                               X_MIN, X_MAX, Y_MIN, Y_MAX);

    cudaEventRecord(stop_evt, 0);
    cudaEventSynchronize(stop_evt);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Fatal Error: CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_image_buffer);
        free(h_image_buffer);
        return EXIT_FAILURE;
    }

    // Calculate elapsed time
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start_evt, stop_evt);
    double elapsed_sec = (double)elapsed_ms / 1000.0;
    printf("GPU Kernel computation completed in %f seconds.\n", elapsed_sec);

    // Copy results from GPU memory to CPU Host memory
    err = cudaMemcpy(h_image_buffer, d_image_buffer, width * height * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Fatal Error: CUDA memcpy Device-to-Host failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_image_buffer);
        free(h_image_buffer);
        return EXIT_FAILURE;
    }

    // Save final PGM image
    save_pgm(output_file, width, height, max_iter, h_image_buffer);

    // Clean up GPU resources
    cudaEventDestroy(start_evt);
    cudaEventDestroy(stop_evt);
    cudaFree(d_image_buffer);

    // Clean up host memory
    free(h_image_buffer);

    return EXIT_SUCCESS;
}
