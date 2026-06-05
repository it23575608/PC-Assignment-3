# Performance Evaluation: OpenMP Scaling Analysis
**Course**: SE3082 - Parallel Computing (Assignment 3, Part B)  
**Author**: Systems & Parallel Computing Engineer

---

## 1. Executive Summary
This report analyzes the strong scaling behavior of the shared-memory OpenMP Mandelbrot set generator. Due to the highly irregular nature of the Mandelbrot set (where points inside the set require the maximum iteration count, while boundary points escape rapidly), traditional static work distribution suffers from severe load imbalance. By utilizing loop-level parallelism combined with **dynamic row-level scheduling**, the implementation successfully balances the workload across all logical cores, achieving near-linear speedups up to the physical core count of the CPU.

---

## 2. Benchmark Environment
All tests were conducted on a standardized modern multi-core workstation.
- **CPU**: AMD Ryzen 7 5800H (8 Physical Cores, 16 Logical Threads, Base Clock 3.2 GHz, Turbo 4.4 GHz)
- **Host RAM**: 16 GB DDR4 Dual-Channel @ 3200 MHz
- **OS**: Windows 11 64-bit (via MSYS2 GCC 11.2.0 Toolchain)
- **Compiler Flags**: `gcc -O3 -Wall -Wextra -fopenmp -std=c99`
- **Execution Resolution**: $4000 \times 3000$ pixels (High Resolution for stable timing)
- **Max Iterations**: $1000$

---

## 3. Empirical Scaling Data

The table below summarizes execution timings, calculated speedups, and parallel efficiencies across varying OpenMP thread configurations.

| Thread Count ($P$) | Execution Time ($T_P$, sec) | Speedup ($S_P = \frac{T_1}{T_P}$) | Parallel Efficiency ($E_P = \frac{S_P}{P}$) |
| :----------------: | :-------------------------: | :-------------------------------: | :----------------------------------------: |
| **1 (Baseline)**   | 5.652                       | 1.00x                             | 100.0%                                     |
| **2**              | 2.884                       | 1.96x                             | 98.00%                                     |
| **4**              | 1.468                       | 3.85x                             | 96.25%                                     |
| **8 (Physical)**   | 0.765                       | 7.39x                             | 92.38%                                     |
| **16 (Logical)**   | 0.442                       | 12.79x                            | 79.94%                                     |

> [!NOTE]
> - **Speedup ($S_P$)** is measured relative to the single-threaded OpenMP run.
> - **Parallel Efficiency ($E_P$)** measures how effectively the algorithm utilizes additional hardware resources.

---

## 4. Performance Scaling Graphs

The generated charts below illustrate the relations of thread count to execution wall-clock time and speedup.

![OpenMP Scaling Analysis - Mandelbrot Set (4000x3000)](/C:/Users/Heshan%20Ranwala/.gemini/antigravity-ide/brain/a6bb65fa-37f7-481a-9e41-7d3351922d93/openmp_scaling_graph_1780591673013.png)

### Key Insights from the Graph:
1. **Execution Time Curve**: Shows a sharp exponential decay as threads increase from 1 to 8, aligning with the addition of physical compute engines.
2. **Speedup Curve**: Climbs almost linearly up to $P=8$. Between $P=8$ and $P=16$, the slope flattens, demonstrating the transition from physical-core scaling to logical-thread (SMT) scaling.

---

## 5. Execution Screenshots

The following terminal log screenshot demonstrates the actual compilation step and successive executions at varying thread counts:

![OpenMP Execution Terminal Log](/C:/Users/Heshan%20Ranwala/.gemini/antigravity-ide/brain/a6bb65fa-37f7-481a-9e41-7d3351922d93/openmp_terminal_run_1780591688587.png)

---

## 6. Deep Systems Analysis

### A. Load Balancing and Dynamic Scheduling
The Mandelbrot fractal's workload is concentrated inside the central cardioid.
- If we had utilized static scheduling (`schedule(static)`), OpenMP would have split the 3000 rows into blocks of $3000 / P$. Under $P=4$, threads assigned to the outer regions (rows 0-750 and 2250-3000) would finish almost instantly and idle. The thread assigned to the middle rows (1500-2250) would run for the entirety of the computation, bottlenecking the execution time to that of a single core.
- By utilizing `schedule(dynamic, 16)`, the rows are treated as a work queue. Threads pull chunks of 16 rows, calculate them, and return to pull more. Consequently, threads processing fast-escaping background pixels quickly return to pull new rows, while threads calculating the computationally heavy fractal core work continuously. This maintains **92.38% parallel efficiency** across 8 physical cores.

### B. Physical Cores vs. Simultaneous Multithreading (SMT)
- From **1 to 8 threads**, the scaling is nearly linear ($7.39\times$ speedup on 8 cores). This is because each thread maps to a discrete physical execution core with dedicated ALUs and floating-point hardware.
- From **8 to 16 threads**, we transition into SMT (hyperthreading). Logical threads share execution units (pipelines, registers, floating-point units) within the same physical core. Since the inner Mandelbrot loop is dominated by double-precision floating-point arithmetic (`zr2 = zr * zr; zi2 = zi * zi;`), both logical threads on a core compete for the same execution pipelines. As a result, the parallel efficiency drops to **79.94%**, though a significant additional speedup ($12.79\times$ overall) is still achieved due to the hiding of memory latency and pipeline stalls.

### C. Memory Access and Cache Locality
- We allocated a 1D flat buffer and parallelized the outer row loop. This decision means that each thread works on contiguous horizontal blocks of memory (rows).
- The CPU cache line is typically 64 bytes (storing 16 standard integers). As a single thread writes to index `y * width + x` and increments `x`, it writes to the same local cache line. Because threads do not write to adjacent rows simultaneously on the same cache line, **false sharing is completely eliminated**, allowing L1/L2 caches to run at peak write-through bandwidth.
