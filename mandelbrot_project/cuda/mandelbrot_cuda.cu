/**
 * @file mandelbrot_cuda.cu
 * @brief Massively Parallel GPU Mandelbrot Implementation using NVIDIA CUDA
 * @course SE3082 - Parallel Computing (Assignment 3)
 * 
 * PARALLEL DESIGN ARCHITECTURE:
 * 1. Massively Parallel Pixel-to-Thread Mapping:
 *    Unlike CPU parallelism which divides work at the row level, CUDA allows us to spawn
 *    millions of threads. We map each individual pixel (x, y) directly to a dedicated CUDA thread.
 *    This represents a fine-grained data parallel approach.
 * 
 * 2. 2D Block and Grid Configuration:
 *    The image space is a 2D grid. We define our thread blocks in 2D:
 *    - `dim3 block_dim(16, 16)`: Spawns 256 threads per block. 16 threads in each dimension
 *      align perfectly with CUDA's warp size of 32 threads.
 *    - `dim3 grid_dim((width + 15) / 16, (height + 15) / 16)`: Automatically calculates the 
 *      number of blocks needed in the X and Y directions to cover the entire resolution,
 *      handling cases where the dimensions are not perfect multiples of 16.
 * 
 * 3. Warp Divergence Mitigation:
 *    A warp (32 threads) executes in lockstep (SIMT). If threads in a warp execute different 
 *    paths (e.g. one thread runs 10 iterations, another runs 1000), the execution serialized, 
 *    causing stalls.
 *    - By grouping threads into 2D blocks, adjacent threads process physically adjacent pixels.
 *      Due to the spatial coherence of the Mandelbrot set, neighboring pixels escape at very 
 *      similar iterations, meaning warp threads are highly likely to escape together, 
 *      minimizing warp divergence.
 * 
 * 4. Error Checking & Resource Staging:
 *    - Device memory is allocated via `cudaMalloc`.
 *    - The kernel is launched asynchronously.
 *    - API calls check for launch and copy errors (`cudaGetLastError`, checking `cudaSuccess`).
 *    - Results are copied from device back to host using `cudaMemcpy(..., cudaMemcpyDeviceToHost)`.
 * 
 * 5. High-Resolution Event Timing:
 *    Rather than using host timers which can include kernel launch queue latency, we use CUDA 
 *    events (`cudaEvent_t`) recorded on the default GPU stream to accurately measure the GPU 
 *    kernel execution time.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Default configuration parameters */
#define DEFAULT_WIDTH 1600
#define DEFAULT_HEIGHT 1200
#define DEFAULT_MAX_ITER 1000
#define DEFAULT_OUTPUT "mandelbrot_cuda.pgm"

/* Define Mandelbrot complex plane coordinate bounds */
#define X_MIN -2.0
#define X_MAX 1.0
#define Y_MIN -1.5
#define Y_MAX 1.5
#define ESCAPE_RADIUS_SQ 4.0

/**
 * @brief CUDA Kernel for computing the Mandelbrot set.
 * Maps each thread to a single pixel coordinate.
 */
__global__ void mandelbrot_kernel(int *d_image_buffer, int width, int height, int max_iter,
                                  double x_min, double x_max, double y_min, double y_max) {
    /* Calculate global 2D pixel coordinates for this thread */
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    /* Boundary check: ensure thread does not process pixels outside the image dimensions */
    if (x < width && y < height) {
        /* Map pixel to complex coordinate (cr, ci) */
        double cr = x_min + (double)x * (x_max - x_min) / (double)width;
        double ci = y_min + (double)y * (y_max - y_min) / (double)height;

        double zr = 0.0;
        double zi = 0.0;
        double zr2 = 0.0;
        double zi2 = 0.0;
        int iter = 0;

        /* Quadratic recurrence calculation: z_{k+1} = z_k^2 + c */
        while (zr2 + zi2 <= ESCAPE_RADIUS_SQ && iter < max_iter) {
            double temp = zr2 - zi2 + cr;
            zi = 2.0 * zr * zi + ci;
            zr = temp;
            
            zr2 = zr * zr;
            zi2 = zi * zi;
            iter++;
        }

        /* Write result to flat 1D device buffer */
        d_image_buffer[y * width + x] = iter;
    }
}

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
    int block_x = 16;
    int block_y = 16;

    /* Parse command line arguments if provided */
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

    /* Allocate host memory for output image */
    int *h_image_buffer = (int *)malloc(width * height * sizeof(int));
    if (!h_image_buffer) {
        fprintf(stderr, "Fatal Error: Host memory allocation failed.\n");
        return EXIT_FAILURE;
    }

    /* Allocate device memory for output image */
    int *d_image_buffer = NULL;
    cudaError_t err = cudaMalloc((void **)&d_image_buffer, width * height * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Fatal Error: Device memory allocation failed: %s\n", cudaGetErrorString(err));
        free(h_image_buffer);
        return EXIT_FAILURE;
    }

    /* Define CUDA execution configuration: 2D Block and 2D Grid */
    dim3 block_dim(block_x, block_y);
    dim3 grid_dim((width + block_dim.x - 1) / block_dim.x, 
                  (height + block_dim.y - 1) / block_dim.y);

    printf("Launching CUDA Kernel with block size %dx%d and grid size %dx%d...\n", 
           block_dim.x, block_dim.y, grid_dim.x, grid_dim.y);

    /* Set up CUDA high-resolution events for profiling */
    cudaEvent_t start_evt, stop_evt;
    cudaEventCreate(&start_evt);
    cudaEventCreate(&stop_evt);

    /* Record start event on default stream */
    cudaEventRecord(start_evt, 0);

    /* Launch Kernel */
    mandelbrot_kernel<<<grid_dim, block_dim>>>(d_image_buffer, width, height, max_iter, 
                                               X_MIN, X_MAX, Y_MIN, Y_MAX);

    /* Record stop event on default stream */
    cudaEventRecord(stop_evt, 0);

    /* Wait for GPU computation to complete */
    cudaEventSynchronize(stop_evt);

    /* Check for kernel execution or configuration launch errors */
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Fatal Error: CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_image_buffer);
        free(h_image_buffer);
        return EXIT_FAILURE;
    }

    /* Calculate elapsed time on GPU */
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start_evt, stop_evt);
    double elapsed_sec = (double)elapsed_ms / 1000.0;
    printf("GPU Kernel computation completed in %f seconds.\n", elapsed_sec);

    /* Copy output buffer from GPU memory to CPU Host memory */
    err = cudaMemcpy(h_image_buffer, d_image_buffer, width * height * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Fatal Error: CUDA memcpy Device-to-Host failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_image_buffer);
        free(h_image_buffer);
        return EXIT_FAILURE;
    }

    /* Save the gathered image buffer from host */
    save_pgm(output_file, width, height, max_iter, h_image_buffer);

    /* Clean up device events and buffers */
    cudaEventDestroy(start_evt);
    cudaEventDestroy(stop_evt);
    cudaFree(d_image_buffer);

    /* Clean up host memory */
    free(h_image_buffer);

    return EXIT_SUCCESS;
}
