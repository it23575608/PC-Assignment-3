/*
 * Mandelbrot set generation using MPI (Master-Worker paradigm)
 * Course: SE3082 - Parallel Computing
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define DEFAULT_WIDTH 1600
#define DEFAULT_HEIGHT 1200
#define DEFAULT_MAX_ITER 1000
#define DEFAULT_OUTPUT "mandelbrot_mpi.pgm"

#define X_MIN -2.0
#define X_MAX 1.0
#define Y_MIN -1.5
#define Y_MAX 1.5
#define ESCAPE_RADIUS_SQ 4.0

// MPI Communication Tags
#define TAG_TASK 1
#define TAG_RESULT 2

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
    int rank, size;

    // Initialize MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int width = DEFAULT_WIDTH;
    int height = DEFAULT_HEIGHT;
    int max_iter = DEFAULT_MAX_ITER;
    const char *output_file = DEFAULT_OUTPUT;

    // Parse command line arguments if provided
    if (argc > 1) width = atoi(argv[1]);
    if (argc > 2) height = atoi(argv[2]);
    if (argc > 3) max_iter = atoi(argv[3]);
    if (argc > 4) output_file = argv[4];

    // Master Process
    if (rank == 0) {
        printf("====================================================\n");
        printf("Mandelbrot Parallel Implementation (MPI Master-Worker)\n");
        printf("Resolution       : %d x %d pixels\n", width, height);
        printf("Max Iterations   : %d\n", max_iter);
        printf("MPI Processes    : %d (1 Master, %d Workers)\n", size, size - 1);
        printf("Output File      : %s\n", output_file);
        printf("====================================================\n");

        // Allocate main image buffer
        int *image_buffer = (int *)malloc(width * height * sizeof(int));
        if (!image_buffer) {
            fprintf(stderr, "Fatal Error: Master failed to allocate memory for image buffer.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return EXIT_FAILURE;
        }

        double start_time = MPI_Wtime();

        // Fallback for single process run (sequential calculation to avoid deadlock)
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
        // Master coordination logic
        else {
            int next_row = 0;
            int active_workers = 0;

            // Send initial task to each worker
            for (int w = 1; w < size; w++) {
                if (next_row < height) {
                    MPI_Send(&next_row, 1, MPI_INT, w, TAG_TASK, MPI_COMM_WORLD);
                    next_row++;
                    active_workers++;
                } else {
                    int term = -1; // Send shutdown signal
                    MPI_Send(&term, 1, MPI_INT, w, TAG_TASK, MPI_COMM_WORLD);
                }
            }

            // Temp buffer to receive worker row outputs: index 0 is row index, 1...width are pixels
            int *recv_buffer = (int *)malloc((width + 1) * sizeof(int));
            if (!recv_buffer) {
                fprintf(stderr, "Fatal Error: Master failed to allocate receive buffer.\n");
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                return EXIT_FAILURE;
            }

            // Collect results and distribute remaining rows
            while (active_workers > 0) {
                MPI_Status status;
                
                MPI_Recv(recv_buffer, width + 1, MPI_INT, MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &status);
                
                int worker_rank = status.MPI_SOURCE;
                int finished_row = recv_buffer[0];

                // Write results to main buffer
                for (int x = 0; x < width; x++) {
                    image_buffer[finished_row * width + x] = recv_buffer[1 + x];
                }

                // If rows remain, send next row, else send shutdown signal
                if (next_row < height) {
                    MPI_Send(&next_row, 1, MPI_INT, worker_rank, TAG_TASK, MPI_COMM_WORLD);
                    next_row++;
                } else {
                    int term = -1;
                    MPI_Send(&term, 1, MPI_INT, worker_rank, TAG_TASK, MPI_COMM_WORLD);
                    active_workers--;
                }
            }

            free(recv_buffer);
        }

        double end_time = MPI_Wtime();
        printf("MPI Master-Worker computation completed in %f seconds.\n", end_time - start_time);

        // Save gathered results
        save_pgm(output_file, width, height, max_iter, image_buffer);

        free(image_buffer);
    }
    // Worker Processes
    else {
        // Allocate buffer to store row metadata + row pixels
        int *row_buffer = (int *)malloc((width + 1) * sizeof(int));
        if (!row_buffer) {
            fprintf(stderr, "Fatal Error: Worker rank %d failed to allocate memory.\n", rank);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return EXIT_FAILURE;
        }

        while (1) {
            int row_index;

            // Receive row index task
            MPI_Recv(&row_index, 1, MPI_INT, 0, TAG_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Exit loop if termination signal is received
            if (row_index == -1) {
                break;
            }

            // Pack row index
            row_buffer[0] = row_index;

            double ci = Y_MIN + (double)row_index * (Y_MAX - Y_MIN) / (double)height;

            // Calculate pixels for this row
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

                // Store iteration value offset by 1
                row_buffer[1 + x] = iter;
            }

            // Send packed row buffer back to master
            MPI_Send(row_buffer, width + 1, MPI_INT, 0, TAG_RESULT, MPI_COMM_WORLD);
        }

        free(row_buffer);
    }

    // Finalize MPI
    MPI_Finalize();
    return EXIT_SUCCESS;
}
