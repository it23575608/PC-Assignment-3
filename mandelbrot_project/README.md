# Mandelbrot Set Parallelization Project (SE3082 - Parallel Computing)
This repository contains parallel implementations of the Mandelbrot set algorithm across different computing paradigms: **Serial (Reference)**, **OpenMP (Shared Memory)**, **MPI (Distributed Memory)**, and **CUDA (Massively Parallel GPU)**.

---

## Phase 1: Serial & OpenMP Implementations

This phase provides the baseline serial implementation and the multi-threaded OpenMP implementation. Both versions generate output as binary `.pgm` (Portable Graymap) files, which enables exact bit-for-bit verification of parallel correctness.

---

## 1. Directory Structure

```
/mandelbrot_project
  ├── /serial/                # Serial Reference Implementation
  │     ├── mandelbrot_serial.c
  │     └── Makefile
  ├── /openmp/                # OpenMP Implementation (Shared Memory)
  │     ├── mandelbrot_openmp.c
  │     └── Makefile
  ├── /mpi/                   # MPI Implementation (Distributed Memory)
  │     ├── mandelbrot_mpi.c
  │     └── Makefile
  ├── /cuda/                  # CUDA Implementation (Massively Parallel GPU)
  │     ├── mandelbrot_cuda.cu
  │     └── Makefile
  └── README.md               # Build and Run Guidelines
```

---

## 2. Compilation and Build Instructions

Both directories contain custom `Makefiles` that compile the sources with standard optimization flags (`-O3 -Wall -Wextra`).

### A. Compiling the Serial Baseline
Navigate to the serial directory and build the executable:
```bash
cd mandelbrot_project/serial
make
```
This generates the executable `mandelbrot_serial` (or `mandelbrot_serial.exe` on Windows).

### B. Compiling the OpenMP Version
Navigate to the openmp directory and build the executable:
```bash
cd ../openmp
make
```
This compile command uses the `-fopenmp` flag to enable parallel loop multi-threading. It generates `mandelbrot_openmp` (or `mandelbrot_openmp.exe` on Windows).

### C. Compiling the MPI Version
Navigate to the mpi directory and build the executable:
```bash
cd ../mpi
make
```
This compile command uses `mpicc` to link the Message Passing Interface library. It generates `mandelbrot_mpi` (or `mandelbrot_mpi.exe` on Windows).

### D. Compiling the CUDA Version
Navigate to the cuda directory and build the executable:
```bash
cd ../cuda
make
```
This compile command uses NVIDIA's `nvcc` compiler to compile GPU device code and host staging routines. It generates `mandelbrot_cuda` (or `mandelbrot_cuda.exe` on Windows).

---

## 3. Running the Programs

Both programs accept the same command-line arguments to allow direct comparison of timings and results:
```bash
./executable [width] [height] [max_iterations] [output_filename.pgm]
```

### Run Parameters (Defaults)
If no arguments are provided, the programs fallback to the following defaults:
- **Width**: `1600`
- **Height**: `1200`
- **Max Iterations**: `1000`
- **Output File**: `mandelbrot_serial.pgm` or `mandelbrot_openmp.pgm`

### Examples

**Run the Serial Baseline:**
```bash
./mandelbrot_serial 1600 1200 1000 mandelbrot_serial.pgm
```

**Run the OpenMP Version (Configuring Threads):**
You can control the number of threads spawned by OpenMP using the `OMP_NUM_THREADS` environment variable.

*On Windows (CMD):*
```cmd
set OMP_NUM_THREADS=4
mandelbrot_openmp.exe 1600 1200 1000 mandelbrot_openmp.pgm
```

*On Windows (PowerShell):*
```powershell
$env:OMP_NUM_THREADS=4
.\mandelbrot_openmp.exe 1600 1200 1000 mandelbrot_openmp.pgm
```

*On Linux/MacOS/Git Bash:*
```bash
OMP_NUM_THREADS=4 ./mandelbrot_openmp 1600 1200 1000 mandelbrot_openmp.pgm
```

**Run the MPI Version (Distributed Memory):**
You run the MPI program using `mpiexec` or `mpirun` specifying the number of processes (where 1 rank is the master and the remaining are calculation workers).

```bash
# Run with 4 processes (1 master, 3 workers)
mpiexec -n 4 ./mandelbrot_mpi 1600 1200 1000 mandelbrot_mpi.pgm
```

**Run the CUDA Version (Massively Parallel GPU):**
You run the compiled CUDA binary directly:

```bash
./mandelbrot_cuda 1600 1200 1000 mandelbrot_cuda.pgm
```

---

## 4. Verification and Correctness Checking

To verify that the parallel OpenMP implementation is 100% correct and mathematically equivalent to the serial implementation:

1. Run both implementations with identical parameters:
   ```bash
   # Serial
   ./mandelbrot_serial 1920 1080 2000 output_serial.pgm
    
   # OpenMP
   OMP_NUM_THREADS=4 ./mandelbrot_openmp 1920 1080 2000 output_openmp.pgm

   # MPI
   mpiexec -n 4 ./mandelbrot_mpi 1920 1080 2000 output_mpi.pgm

   # CUDA
   ./mandelbrot_cuda 1920 1080 2000 output_cuda.pgm
   ```

2. Compare the output binary PGM files. They must be bit-for-bit identical.

   *On Windows:*
   ```cmd
   fc /b output_serial.pgm output_openmp.pgm
   fc /b output_serial.pgm output_mpi.pgm
   fc /b output_serial.pgm output_cuda.pgm
   ```
   Expected output: `FC: no differences encountered`

   *On Linux / macOS / Git Bash:*
   ```bash
   diff output_serial.pgm output_openmp.pgm
   diff output_serial.pgm output_mpi.pgm
   diff output_serial.pgm output_cuda.pgm
   ```
   Expected output: (No console output, indicating the files match perfectly) or you can check using md5sum:
   ```bash
   md5sum output_serial.pgm output_openmp.pgm output_mpi.pgm output_cuda.pgm
   ```
