# Complete Build and Execution Guide

This guide provides step-by-step instructions for building binaries from scratch and generating CSV performance metrics for GPU benchmarking.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start (Recommended)](#quick-start-recommended)
3. [Understanding the Binaries](#understanding-the-binaries)
4. [Manual Build Process](#manual-build-process)
5. [Step-by-Step Execution Workflow](#step-by-step-execution-workflow)
6. [Understanding the Output CSVs](#understanding-the-output-csvs)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

1. **NVIDIA CUDA Toolkit** (with nvcc compiler)
   - Verify installation: `nvcc --version`
   - Configured for compute capability `sm_75` (RTX 2080 Ti)

2. **cuBLAS Library** (usually included with CUDA Toolkit)
   - Required for GEMM compute benchmarks

3. **Python 3** with standard library
   - Verify installation: `python3 --version`
   - Required modules: csv, statistics, json, glob, re (all standard library)

4. **Bash shell**
   - For running orchestration scripts

5. **nvidia-smi**
   - For GPU device detection
   - Verify: `nvidia-smi`

### Hardware Requirements

- NVIDIA GPU (configured for RTX 2080 Ti with compute capability 7.5)
- Sufficient GPU memory for benchmarks (typically 4GB+)

---

## Quick Start (Recommended)

For a complete end-to-end run (build everything + generate all CSVs):

```bash
cd gpu-perf
bash scripts/run_2080ti_full.sh
```

This single command will:
- Build all binaries
- Run calibration benchmarks
- Execute 16 kernels × 10 trials each (160 total runs)
- Generate final CSV: `data/runs_2080ti_final.csv`

**Estimated time:** 5-15 minutes depending on GPU and trial count

---

## Understanding the Binaries

The project builds **four main binaries**:

### 1. Calibration Binaries (in `bin/`)

#### `bin/props`
- **Source:** `calibration/props.cu`
- **Purpose:** Queries and outputs GPU device properties
- **Output:** Device name, compute capability, SM count, memory specs, warp size, etc.

#### `bin/stream_like`
- **Source:** `calibration/stream_like.cu`
- **Purpose:** Measures sustained memory bandwidth
- **Method:** STREAM-like triad kernel (`a[i] = b[i] + scalar * c[i]`)
- **Test sizes:** 128M, 64M, 32M elements
- **Output:** Peak bandwidth in GB/s

#### `bin/gemm_cublas`
- **Source:** `calibration/gemm_cublas.cu`
- **Purpose:** Measures sustained compute throughput
- **Method:** cuBLAS matrix multiplication (DGEMM)
- **Test sizes:** 8192×8192, 6144×6144, 4096×4096 matrices
- **Output:** Peak performance in GFLOPS

### 2. Main Runner Binary

#### `bin/runner`
- **Source:** `runner/main.cu`
- **Purpose:** Single unified binary that can execute any of 22 CUDA kernels
- **Kernels included:**
  - Memory streaming: vector_add, saxpy
  - Strided access: strided_copy_4, strided_copy_8
  - Transpose: naive_transpose, shared_transpose
  - Matrix multiply: matmul_naive, matmul_tiled
  - Reductions: reduce_sum, reduce_warp, dot_product
  - Convolutions: conv2d_3x3, conv2d_5x5, conv2d_7x7, stencil2d_5pt
  - Atomics: histogram, scatter_atomic, atomic_hotspot
  - Irregular: random_access, vector_add_divergent
  - Shared memory: shared_bank_conflict, scan_block

---

## Manual Build Process

### Step 1: Create Output Directories

```bash
cd gpu-perf
mkdir -p bin data
```

### Step 2: Build Calibration Binaries

```bash
# Build GPU properties detector
nvcc -O3 -arch=sm_75 -o bin/props calibration/props.cu

# Build memory bandwidth benchmark
nvcc -O3 -arch=sm_75 -o bin/stream_like calibration/stream_like.cu

# Build compute throughput benchmark (requires cuBLAS)
nvcc -O3 -arch=sm_75 -lcublas -o bin/gemm_cublas calibration/gemm_cublas.cu
```

**Compiler flags explained:**
- `-O3`: Maximum optimization level
- `-arch=sm_75`: Target compute capability 7.5 (RTX 2080 Ti)
- `-lcublas`: Link cuBLAS library

### Step 3: Build Main Runner Binary

```bash
nvcc -std=c++14 -O3 --ptxas-options=-v -lineinfo -arch=sm_75 -DTILE=32 \
  -o bin/runner runner/main.cu 2> data/ptxas_2080ti.log
```

**Compiler flags explained:**
- `-std=c++14`: Use C++14 standard
- `--ptxas-options=-v`: Verbose PTX assembly information (captures register usage)
- `-lineinfo`: Include debugging line information
- `-DTILE=32`: Define tile size for tiled matrix multiplication
- `2> data/ptxas_2080ti.log`: Capture PTX statistics to log file

---

## Step-by-Step Execution Workflow

### Overview

The complete workflow has **9 steps**:

```
Build → Calibrate → Run Trials → Aggregate → Normalize →
Add Static Counts → Enrich with GPU Metrics → Add Baseline → Final CSV
```

### Step 0: Clean Previous Runs (Optional)

```bash
rm -f bin/* data/trials_*.csv data/runs_*.csv
```

### Step 1: Collect GPU Calibration Data

```bash
# Collect GPU properties
bin/props > data/props_2080ti.out

# Measure sustained memory bandwidth
bin/stream_like > data/stream_like_2080ti.out

# Measure sustained compute throughput
bin/gemm_cublas > data/gemm_cublas_2080ti.out
```

**Expected outputs:**
- `props_2080ti.out`: GPU specifications (1 SM count line, 1 L2 cache line, etc.)
- `stream_like_2080ti.out`: Line starting with "SUSTAINED" showing peak GB/s
- `gemm_cublas_2080ti.out`: Line starting with "SUSTAINED" showing peak GFLOPS

### Step 2: Run Individual Kernel Trials

The runner binary accepts kernel-specific arguments:

```bash
bin/runner --kernel vector_add --N 1048576 --block 256 --warmup 20 --reps 100
```

**Output format:**
```
KERNEL=vector_add N=1048576 block=256 grid=4096 time_ms=0.012345
```

**Common arguments:**
- `--kernel <name>`: Kernel to execute
- `--N <size>`: Problem size (for 1D kernels)
- `--matN <size>`: Matrix dimension (for 2D kernels)
- `--H <height> --W <width>`: Image dimensions (for convolutions)
- `--block <size>`: Thread block size
- `--warmup <n>`: Warmup iterations
- `--reps <n>`: Measurement iterations

### Step 3: Collect Multiple Trials per Kernel

Use the trial collection script to run each kernel multiple times:

```bash
scripts/run_trials.sh <kernel> "<runner_args>" <regs> <shmem> <trials>
```

**Example:**
```bash
scripts/run_trials.sh vector_add \
  "--rows 1048576 --cols 1 --block 256 --warmup 20 --reps 100" \
  12 0 10
```

This runs the kernel 10 times and appends results to `data/trials_vector_add__2080ti.csv`.

**CSV format:**
```
kernel,args,device_name,block,grid,regs,shmem,time_ms
vector_add,"--rows 1048576 --cols 1 --block 256",GeForce RTX 2080 Ti,256,4096,12,0,0.012345
```

**The full pipeline runs 16 kernels** (see `scripts/run_2080ti_full.sh` for complete list):
- vector_add, saxpy, strided_copy_8
- naive_transpose, shared_transpose
- matmul_naive, matmul_tiled
- reduce_sum, dot_product
- histogram, conv2d_3x3, conv2d_7x7
- random_access, vector_add_divergent
- shared_bank_conflict, atomic_hotspot

**Total trials:** 16 kernels × 10 trials = 160 runs

### Step 4: Aggregate Trial Data

```bash
python3 scripts/aggregate_trials.py "data/trials_*__2080ti.csv" > data/runs_2080ti.csv
```

**What this does:**
- Groups trials by (kernel, args, device, block, grid, regs, shmem)
- Computes mean and standard deviation of timing measurements
- Outputs single CSV with statistical summary

**Output columns:**
```
kernel, args, device_name, block, grid, regs, shmem, trials, mean_ms, std_ms
```

### Step 5: Normalize Problem Sizes

```bash
python3 scripts/normalize_sizes.py data/runs_2080ti.csv data/runs_2080ti_norm.csv
```

**What this does:**
- Parses various size formats from args: `--N`, `--rows/cols`, `--matN`, `--H/W`
- Converts to unified representation: rows, cols, iters
- Adds these as new columns to CSV

**New columns added:**
```
rows, cols, iters
```

### Step 6: Add Static Operation Counts

```bash
python3 scripts/static_counts.py \
  data/runs_2080ti_norm.csv \
  data/runs_2080ti_with_counts.csv
```

**What this does:**
- Calculates FLOPs (floating-point operations) per kernel
- Calculates BYTES (memory traffic) per kernel
- Computes arithmetic intensity (FLOPs/BYTES)
- Adds memory access pattern classifications
- Computes working set sizes

**New columns added:**
```
FLOPs, BYTES, arithmetic_intensity, working_set_bytes, mem_pattern
```

**Memory patterns:**
- coalesced: Sequential memory access (vector_add, saxpy)
- strided: Non-unit stride access (strided_copy_8)
- shared: Shared memory usage (matmul_tiled, shared_transpose)
- atomic: Atomic operations (histogram, atomic_hotspot)
- irregular: Random access (random_access)

### Step 7: Enrich with GPU-Specific Metrics

```bash
python3 scripts/enrich_with_gpu_metrics.py \
  data/runs_2080ti_with_counts.csv \
  data/props_2080ti.out \
  data/stream_like_2080ti.out \
  data/gemm_cublas_2080ti.out \
  data/runs_2080ti_enriched.csv
```

**What this does:**
- Parses GPU properties from calibration outputs
- Adds device specifications to each row
- Adds sustained performance ceilings (bandwidth and compute)

**New columns added:**
```
cc_major, cc_minor, sm_count, l2_cache_bytes, warp_size,
sustained_bw_GBps, sustained_gflops, roofline_bound
```

**Roofline bound determination:**
- "memory": Performance limited by memory bandwidth (low arithmetic intensity)
- "compute": Performance limited by compute throughput (high arithmetic intensity)

### Step 8: Add Single-Thread Baseline Model

```bash
python3 scripts/add_singlethread_baseline.py \
  data/runs_2080ti_enriched.csv \
  data/device_calibration_2080ti.json \
  data/runs_2080ti_final.csv \
  32
```

**What this does:**
- Models theoretical single-thread execution time (T1) using roofline model
- T1 = max(BYTES/bandwidth, FLOPs/compute_rate)
- Calculates theoretical speedup: T1 / T_parallel
- Uses warp size (32) as parallelism unit

**New columns added:**
```
T1_model_ms, speedup_model
```

### Step 9: Final Output

**Result:** `data/runs_2080ti_final.csv`

**Complete column set (43 columns):**
- Kernel identity: kernel, args, device_name
- Launch config: block, grid, regs, shmem
- Timing stats: trials, mean_ms, std_ms
- Problem size: rows, cols, iters
- Operation counts: FLOPs, BYTES, arithmetic_intensity, working_set_bytes
- GPU metrics: cc_major, cc_minor, sm_count, l2_cache_bytes, warp_size
- Performance ceilings: sustained_bw_GBps, sustained_gflops, roofline_bound
- Memory pattern: mem_pattern
- Baseline model: T1_model_ms, speedup_model

---

## Understanding the Output CSVs

### Intermediate CSVs

| File | Rows | Key Columns | Purpose |
|------|------|-------------|---------|
| `trials_<kernel>__2080ti.csv` | ~10 per kernel | kernel, time_ms | Raw trial data |
| `runs_2080ti.csv` | ~16 (one per kernel) | mean_ms, std_ms | Aggregated statistics |
| `runs_2080ti_norm.csv` | ~16 | rows, cols, iters | Normalized sizes |
| `runs_2080ti_with_counts.csv` | ~16 | FLOPs, BYTES, AI | Operation counts |
| `runs_2080ti_enriched.csv` | ~16 | sm_count, sustained_* | GPU-specific metrics |

### Final CSV: `runs_2080ti_final.csv`

**Row structure:**
- One row per kernel configuration
- Typically 16 rows for the default benchmark suite

**Key metrics explained:**

- **mean_ms**: Average execution time across trials (milliseconds)
- **std_ms**: Standard deviation of execution time
- **FLOPs**: Total floating-point operations for the problem size
- **BYTES**: Total memory traffic (bytes read + bytes written)
- **arithmetic_intensity**: FLOPs per byte (FLOPs/BYTES) - higher = more compute-bound
- **sustained_bw_GBps**: Peak memory bandwidth measured on this GPU
- **sustained_gflops**: Peak compute throughput measured on this GPU
- **roofline_bound**: Whether kernel is "memory" or "compute" bound
- **T1_model_ms**: Theoretical single-thread execution time
- **speedup_model**: Theoretical speedup (T1/T_parallel)

**Example row interpretation:**

```csv
kernel=vector_add, mean_ms=0.012, FLOPs=1048576, BYTES=12582912,
arithmetic_intensity=0.083, roofline_bound=memory, speedup_model=2340.5
```

This tells us:
- Vector addition took 0.012 ms on average
- Low arithmetic intensity (0.083 FLOPs/byte) → memory-bound
- Achieved 2340× speedup vs. theoretical single-thread execution

---

## Troubleshooting

### Build Issues

**Problem:** `nvcc: command not found`
```bash
# Solution: Add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Problem:** `cannot find -lcublas`
```bash
# Solution: Verify cuBLAS installation
ls /usr/local/cuda/lib64/libcublas*
# If missing, reinstall CUDA Toolkit with cuBLAS
```

**Problem:** Architecture mismatch errors
```bash
# Solution: Change -arch flag to match your GPU
# RTX 3090: -arch=sm_86
# RTX 4090: -arch=sm_89
# V100: -arch=sm_70
# A100: -arch=sm_80
```

### Runtime Issues

**Problem:** No GPU detected
```bash
# Check GPU visibility
nvidia-smi
# Check CUDA runtime
bin/props
```

**Problem:** CUDA out of memory
```bash
# Solution: Reduce problem sizes in run_2080ti_full.sh
# Edit kernel arguments, e.g., change --N 8388608 to --N 4194304
```

**Problem:** Trials CSV is empty
```bash
# Check runner output manually
bin/runner --kernel vector_add --N 1048576 --block 256 --warmup 20 --reps 100
# Verify output format matches expected pattern
```

### CSV Generation Issues

**Problem:** Missing FLOPs/BYTES columns
```bash
# Ensure static_counts.py recognizes your kernel
# Add kernel case to scripts/static_counts.py if using custom kernels
```

**Problem:** Missing SUSTAINED lines in calibration output
```bash
# Re-run calibration binaries
bin/stream_like
bin/gemm_cublas
# Check for lines starting with "SUSTAINED"
```

**Problem:** Python script failures
```bash
# Verify Python 3 installation
python3 --version
# Check all required CSV files exist
ls data/trials_*__2080ti.csv
ls data/props_2080ti.out
```

---

## Advanced Usage

### Running a Single Kernel

```bash
# Build runner
nvcc -std=c++14 -O3 -arch=sm_75 -DTILE=32 -o bin/runner runner/main.cu

# Run single trial
bin/runner --kernel matmul_tiled --matN 2048 --block 256 --warmup 10 --reps 50

# Collect 10 trials
scripts/run_trials.sh matmul_tiled \
  "--rows 2048 --cols 2048 --block 256 --warmup 10 --reps 50" \
  32 2048 10
```

### Adding Custom Kernels

1. Create kernel header in `kernels/my_kernel.cuh`
2. Add kernel function and wrapper
3. Include header in `runner/main.cu`
4. Add case to kernel dispatcher
5. Update `static_counts.py` for FLOPs/BYTES calculation
6. Rebuild and run trials

### Changing Target GPU

Edit all scripts to replace `2080ti` with your device identifier:
```bash
# Example for RTX 3090
sed -i 's/2080ti/3090/g' scripts/run_2080ti_full.sh
sed -i 's/sm_75/sm_86/g' scripts/run_2080ti_full.sh
```

---

## Summary

**Full pipeline command:**
```bash
bash scripts/run_2080ti_full.sh
```

**Result:**
- `data/runs_2080ti_final.csv` with 43 columns and ~16 rows
- Complete performance characterization of 16 CUDA kernels
- Ready for analysis, visualization, or machine learning

**Typical use cases:**
- GPU performance analysis
- Roofline modeling
- Kernel optimization studies
- Machine learning training data for performance prediction
- Teaching GPU programming concepts

---

## Additional Resources

- **Kernel source code:** `kernels/*.cuh`
- **PTX statistics:** `data/ptxas_2080ti.log`
- **Raw trial data:** `data/trials_*__2080ti.csv`
- **Calibration data:** `data/props_2080ti.out`, `data/stream_like_2080ti.out`, `data/gemm_cublas_2080ti.out`
