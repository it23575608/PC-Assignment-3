/**
 * @file mandelbrot_mpi.c
 * @brief Distributed-Memory Parallel Mandelbrot Implementation using MPI
 * @course SE3082 - Parallel Computing (Assignment 3)
 * 
 * PARALLEL DESIGN ARCHITECTURE:
 * 1. Distributed-Memory Master-Worker Paradigm:
 *    Under MPI, processes do not share physical memory. To handle the irregular 
 *    workload of the Mandelbrot set, we implement a dynamic Master-Worker framework:
 *    - Rank 0 (Master) coordinates the computation. It maintains a global image buffer,
 *      assigns rows to workers dynamically, receives results, and writes them into the image.
 *    - Ranks 1 to (size-1) (Workers) act as processing nodes. They request/receive row
 *      indices from the Master, compute the iteration values for that row, and send the
 *      data back.
 * 
 * 2. Communication Optimization (Packed Messages):
 *    To minimize network round-trip times and latency:
 *    - The worker packs both the metadata and row data into a single 1D array of size
 *      (width + 1). The first element (index 0) stores the 'row_index' being returned.
 *      Indices 1 to 'width' store the iteration count array for that row.
 *    - This allows the worker to send the entire computation result back to the Master
 *      using a single MPI_Send message, rather than sending a separate row-header message followed
 *      by the payload.
 * 
 * 3. Tag Definitions:
 *    - TAG_TASK (1): Master sends a row index to a worker. If the index is -1, it indicates termination.
 *    - TAG_RESULT (2): Worker sends the calculated row data (packed format) to the Master.
 * 
 * 4. Resiliency & Single-Process Fallback:
 *    If the MPI environment is launched with only 1 process (size == 1), the program falls back
 *    to computing the set sequentially on Rank 0, preventing deadlocks that would occur when
 *    waiting for workers that do not exist.
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/* Default configuration parameters */
#define DEFAULT_WIDTH 1600
#define DEFAULT_HEIGHT 1200
#define DEFAULT_MAX_ITER 1000
#define DEFAULT_OUTPUT "mandelbrot_mpi.pgm"

/* Define Mandelbrot complex plane coordinate bounds */
#define X_MIN -2.0
#define X_MAX 1.0
#define Y_MIN -1.5
#define Y_MAX 1.5
#define ESCAPE_RADIUS_SQ 4.0

/* MPI Communication Tags */
#define TAG_TASK 1
#define TAG_RESULT 2

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
    int rank, size;

    /* Initialize MPI environment */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int width = DEFAULT_WIDTH;
    int height = DEFAULT_HEIGHT;
    int max_iter = DEFAULT_MAX_ITER;
    const char *output_file = DEFAULT_OUTPUT;

    /* Parse command line arguments if provided */
    if (argc > 1) width = atoi(argv[1]);
    if (argc > 2) height = atoi(argv[2]);
    if (argc > 3) max_iter = atoi(argv[3]);
    if (argc > 4) output_file = argv[4];

    /* =========================================================================
     * MASTER PROCESS CODE (RANK 0)
     * ========================================================================= */
    if (rank == 0) {
        printf("====================================================\n");
        printf("Mandelbrot Parallel Implementation (MPI Master-Worker)\n");
        printf("Resolution       : %d x %d pixels\n", width, height);
        printf("Max Iterations   : %d\n", max_iter);
        printf("MPI Processes    : %d (1 Master, %d Workers)\n", size, size - 1);
        printf("Output File      : %s\n", output_file);
        printf("====================================================\n");

        /* Allocate the full flat image buffer on Master */
        int *image_buffer = (int *)malloc(width * height * sizeof(int));
        if (!image_buffer) {
            fprintf(stderr, "Fatal Error: Master failed to allocate memory for image buffer.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return EXIT_FAILURE;
        }

        double start_time = MPI_Wtime();

        /* FALLBACK: If size == 1, compute sequentially to prevent deadlock */
        if (size == 1) {
            printf("Warning: Only 1 MPI process allocated. Running sequential fallback...\n");
            for (int y = 0; y < height; y++) {
                double ci = Y_MIN + (double)y * (Y_MAX - Y_MIN) / (double)height;
                for (int x = 0; x < width; x++) {
                    double cr = X_MIN + (double)x * (X_MAX - X_MIN) / (double)width;
                    double zr = 0.0, zi = 0.0, zr2 = 0.0, zi2 = 0.0;
                    int iter = 0;
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
        } 
        /* MASTER-WORKER COORDINATION */
        else {
            int next_row = 0;
            int active_workers = 0;

            /* 1. Distribute initial task (one row index) to each worker */
            for (int w = 1; w < size; w++) {
                if (next_row < height) {
                    MPI_Send(&next_row, 1, MPI_INT, w, TAG_TASK, MPI_COMM_WORLD);
                    next_row++;
                    active_workers++;
                } else {
                    /* If there are more workers than rows, send termination signal immediately */
                    int term = -1;
                    MPI_Send(&term, 1, MPI_INT, w, TAG_TASK, MPI_COMM_WORLD);
                }
            }

            /* Allocate temporary receive buffer for packed worker results:
             * Index 0: row index
             * Indices 1 ... width: pixel iteration counts
             */
            int *recv_buffer = (int *)malloc((width + 1) * sizeof(int));
            if (!recv_buffer) {
                fprintf(stderr, "Fatal Error: Master failed to allocate receive buffer.\n");
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                return EXIT_FAILURE;
            }

            /* 2. Process incoming worker results and dispatch remaining tasks */
            while (active_workers > 0) {
                MPI_Status status;
                
                /* Wait for any worker to finish and send back a computed row */
                MPI_Recv(recv_buffer, width + 1, MPI_INT, MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &status);
                
                int worker_rank = status.MPI_SOURCE;
                int finished_row = recv_buffer[0];

                /* Copy the received row data into the global image buffer */
                for (int x = 0; x < width; x++) {
                    image_buffer[finished_row * width + x] = recv_buffer[1 + x];
                }

                /* If there are more rows, send the next row to the worker */
                if (next_row < height) {
                    MPI_Send(&next_row, 1, MPI_INT, worker_rank, TAG_TASK, MPI_COMM_WORLD);
                    next_row++;
                } 
                /* Otherwise, send termination signal to free the worker */
                else {
                    int term = -1;
                    MPI_Send(&term, 1, MPI_INT, worker_rank, TAG_TASK, MPI_COMM_WORLD);
                    active_workers--;
                }
            }

            free(recv_buffer);
        }

        double end_time = MPI_Wtime();
        printf("MPI Master-Worker computation completed in %f seconds.\n", end_time - start_time);

        /* Save final gathered image */
        save_pgm(output_file, width, height, max_iter, image_buffer);

        /* Clean up master memory resources */
        free(image_buffer);
    }
    /* =========================================================================
     * WORKER PROCESS CODE (RANKS > 0)
     * ========================================================================= */
    else {
        /* Allocate a row buffer to calculate and pack data for transfer
         * Index 0: stores the row index being returned
         * Indices 1 ... width: stores computed iteration values
         */
        int *row_buffer = (int *)malloc((width + 1) * sizeof(int));
        if (!row_buffer) {
            fprintf(stderr, "Fatal Error: Worker rank %d failed to allocate memory.\n", rank);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return EXIT_FAILURE;
        }

        while (1) {
            int row_index;

            /* Receive row assignment from Master */
            MPI_Recv(&row_index, 1, MPI_INT, 0, TAG_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            /* Check for termination signal */
            if (row_index == -1) {
                break;
            }

            /* Record the row index as metadata in the first element of the buffer */
            row_buffer[0] = row_index;

            /* Compute the Mandelbrot set for the assigned row */
            double ci = Y_MIN + (double)row_index * (Y_MAX - Y_MIN) / (double)height;

            for (int x = 0; x < width; x++) {
                double cr = X_MIN + (double)x * (X_MAX - X_MIN) / (double)width;
                double zr = 0.0;
                double zi = 0.0;
                double zr2 = 0.0;
                double zi2 = 0.0;
                int iter = 0;

                while (zr2 + zi2 <= ESCAPE_RADIUS_SQ && iter < max_iter) {
                    double temp = zr2 - zi2 + cr;
                    zi = 2.0 * zr * zi + ci;
                    zr = temp;
                    zr2 = zr * zr;
                    zi2 = zi * zi;
                    iter++;
                }

                /* Pack the iteration count into the row buffer (offset by 1) */
                row_buffer[1 + x] = iter;
            }

            /* Send the packed row back to Master */
            MPI_Send(row_buffer, width + 1, MPI_INT, 0, TAG_RESULT, MPI_COMM_WORLD);
        }

        /* Clean up worker memory resources */
        free(row_buffer);
    }

    /* Finalize MPI environment */
    MPI_Finalize();
    return EXIT_SUCCESS;
}
