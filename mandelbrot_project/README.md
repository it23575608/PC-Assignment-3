# Mandelbrot Set Parallelization Project (SE3082 - Parallel Computing)

This repository contains parallel implementations of the Mandelbrot set algorithm across different computing paradigms: **Serial (Reference)**, **OpenMP (Shared Memory)**, **MPI (Distributed Memory)**, and **CUDA (Massively Parallel GPU)**.

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
  └── README.md               # Build, Setup, and Run Guidelines
```

---

## 2. Environment Setup

### A. On Ubuntu Server (Linux)
Run the following commands to install GCC (with OpenMP support), OpenMPI, and the CUDA compiler:

```bash
# Update package lists
sudo apt update

# Install GCC, Make, and OpenMP headers
sudo apt install -y build-essential libomp-dev

# Install OpenMPI (Distributed memory environment)
sudo apt install -y openmpi-bin openmpi-common libopenmpi-dev

# Install NVIDIA CUDA Toolkit (Pre-requisite: Server must have an NVIDIA GPU)
sudo apt install -y nvidia-cuda-toolkit
```

### B. On Windows
To build and run these programs on Windows, configure the following:
1. **GCC & OpenMP**: Install [MSYS2](https://www.msys2.org/) and run `pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-make` to install GCC and Make. Add `C:\msys64\mingw64\bin` to your system environment variables `PATH`.
2. **MPI**: Download and install both the **Microsoft MPI Redistributable** (`msmpisetup.exe`) and the **Microsoft MPI SDK** (`msmpisdk.msi`) from the official [Microsoft MS-MPI downloads](https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi).
3. **CUDA**: Install the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (ensure you have an NVIDIA GPU and matching drivers installed).

---

## 3. Compilation and Build Instructions

Each directory contains a custom `Makefile` supporting compilation with optimal optimization flags (`-O3 -Wall -Wextra`).

### A. Compiling the Serial Baseline
```bash
cd serial
make
cd ..
```
*Output binary:* `serial/mandelbrot_serial` (or `mandelbrot_serial.exe` on Windows).

### B. Compiling the OpenMP Version
```bash
cd openmp
make
cd ..
```
*Output binary:* `openmp/mandelbrot_openmp` (or `mandelbrot_openmp.exe` on Windows).

### C. Compiling the MPI Version
```bash
cd mpi
make
cd ..
```
*Output binary:* `mpi/mandelbrot_mpi` (or `mandelbrot_mpi.exe` on Windows).

### D. Compiling the CUDA Version
```bash
cd cuda
make
cd ..
```
*Output binary:* `cuda/mandelbrot_cuda` (or `mandelbrot_cuda.exe` on Windows).

---

## 4. Execution Guidelines (Varying Configuration Parameters)

All implementations share a standardized syntax structure:
```bash
./executable [width] [height] [max_iterations] [output_filename.pgm]
```
*Default params (if none provided):* Width = `1600`, Height = `1200`, Iterations = `1000`.

### A. Running the Serial Baseline
Execute sequentially on a single thread:
```bash
./serial/mandelbrot_serial 4000 3000 1000 mandelbrot_serial.pgm
```

### B. Running OpenMP (Varying Thread Counts)
Control the number of active OpenMP worker threads by setting the `OMP_NUM_THREADS` environment variable prefix before launching:

* **On Linux / macOS:**
  ```bash
  # 1 Thread
  OMP_NUM_THREADS=1 ./openmp/mandelbrot_openmp 4000 3000 1000 omp_t1.pgm
  
  # 2 Threads
  OMP_NUM_THREADS=2 ./openmp/mandelbrot_openmp 4000 3000 1000 omp_t2.pgm
  
  # 4 Threads
  OMP_NUM_THREADS=4 ./openmp/mandelbrot_openmp 4000 3000 1000 omp_t4.pgm
  
  # 8 Threads
  OMP_NUM_THREADS=8 ./openmp/mandelbrot_openmp 4000 3000 1000 omp_t8.pgm
  
  # 16 Threads (SMT logical threads)
  OMP_NUM_THREADS=16 ./openmp/mandelbrot_openmp 4000 3000 1000 omp_t16.pgm
  ```

* **On Windows (PowerShell):**
  ```powershell
  $env:OMP_NUM_THREADS=1; .\openmp\mandelbrot_openmp.exe 4000 3000 1000 omp_t1.pgm
  $env:OMP_NUM_THREADS=2; .\openmp\mandelbrot_openmp.exe 4000 3000 1000 omp_t2.pgm
  $env:OMP_NUM_THREADS=4; .\openmp\mandelbrot_openmp.exe 4000 3000 1000 omp_t4.pgm
  $env:OMP_NUM_THREADS=8; .\openmp\mandelbrot_openmp.exe 4000 3000 1000 omp_t8.pgm
  $env:OMP_NUM_THREADS=16; .\openmp\mandelbrot_openmp.exe 4000 3000 1000 omp_t16.pgm
  ```

---

### C. Running MPI (Varying Process/Rank Counts)
Run using `mpirun` or `mpiexec` specifying the rank size with `-np`. 

> [!IMPORTANT]
> **Oversubscription Check**: If you launch more processes than the physical cores available on your testing machine (e.g. running 16 ranks on a 4-core CPU), Linux OpenMPI will error out. To prevent this, you **must** include the `--oversubscribe` flag.

* **On Linux / macOS:**
  ```bash
  # 1 Process (Sequential fallback)
  mpirun -np 1 ./mpi/mandelbrot_mpi 4000 3000 1000 mpi_p1.pgm
  
  # 2 Processes (1 Master, 1 Worker)
  mpirun --oversubscribe -np 2 ./mpi/mandelbrot_mpi 4000 3000 1000 mpi_p2.pgm
  
  # 4 Processes (1 Master, 3 Workers)
  mpirun --oversubscribe -np 4 ./mpi/mandelbrot_mpi 4000 3000 1000 mpi_p4.pgm
  
  # 8 Processes (1 Master, 7 Workers)
  mpirun --oversubscribe -np 8 ./mpi/mandelbrot_mpi 4000 3000 1000 mpi_p8.pgm
  
  # 16 Processes (1 Master, 15 Workers)
  mpirun --oversubscribe -np 16 ./mpi/mandelbrot_mpi 4000 3000 1000 mpi_p16.pgm
  ```

* **On Windows (PowerShell - MS-MPI allows oversubscription by default):**
  ```powershell
  mpiexec -n 1 .\mpi\mandelbrot_mpi.exe 4000 3000 1000 mpi_p1.pgm
  mpiexec -n 2 .\mpi\mandelbrot_mpi.exe 4000 3000 1000 mpi_p2.pgm
  mpiexec -n 4 .\mpi\mandelbrot_mpi.exe 4000 3000 1000 mpi_p4.pgm
  mpiexec -n 8 .\mpi\mandelbrot_mpi.exe 4000 3000 1000 mpi_p8.pgm
  mpiexec -n 16 .\mpi\mandelbrot_mpi.exe 4000 3000 1000 mpi_p16.pgm
  ```

---

### D. Running CUDA (Varying Block Dimensions)
The CUDA implementation accepts dynamic block dimensions X and Y as the **5th and 6th command-line arguments**.

```bash
# Syntax: ./executable [width] [height] [iterations] [output_file] [block_x] [block_y]
```

* **Run Configurations:**
  ```bash
  # 8x8 Block Layout (64 threads per block)
  ./cuda/mandelbrot_cuda 4000 3000 1000 cuda_8x8.pgm 8 8
  
  # 16x16 Block Layout (256 threads per block - Hardware sweet spot)
  ./cuda/mandelbrot_cuda 4000 3000 1000 cuda_16x16.pgm 16 16
  
  # 32x32 Block Layout (1024 threads per block - Maximum CUDA hardware limit)
  ./cuda/mandelbrot_cuda 4000 3000 1000 cuda_32x32.pgm 32 32
  
  # 16x8 Block Layout (128 threads per block)
  ./cuda/mandelbrot_cuda 4000 3000 1000 cuda_16x8.pgm 16 8
  
  # 8x16 Block Layout (128 threads per block)
  ./cuda/mandelbrot_cuda 4000 3000 1000 cuda_8x16.pgm 8 16
  
  # 32x8 Block Layout (256 threads per block)
  ./cuda/mandelbrot_cuda 4000 3000 1000 cuda_32x8.pgm 32 8
  ```

---

## 5. Correctness and Output Verification

To verify that the parallel OpenMP, MPI, and CUDA implementations are mathematically correct and functionally identical to the reference sequential baseline:

1. Run the benchmarks with the same parameters (e.g. resolution $4000 \times 3000$ and $1000$ iterations).
2. Compare the output binary PGM files using a byte-by-byte comparison utility.

* **On Windows (PowerShell / Command Prompt):**
  ```cmd
  fc /b mandelbrot_serial.pgm omp_t16.pgm
  fc /b mandelbrot_serial.pgm mpi_p16.pgm
  fc /b mandelbrot_serial.pgm cuda_16x16.pgm
  ```
  *Expected Output:* `FC: no differences encountered`

* **On Linux / macOS (Terminal):**
  ```bash
  diff mandelbrot_serial.pgm omp_t16.pgm
  diff mandelbrot_serial.pgm mpi_p16.pgm
  diff mandelbrot_serial.pgm cuda_16x16.pgm
  ```
  *Expected Output:* (No text output, representing a perfect match) or verify via checksums:
  ```bash
  md5sum mandelbrot_serial.pgm omp_t16.pgm mpi_p16.pgm cuda_16x16.pgm
  ```
  *Expected Output:* Identical hash strings for all files.
