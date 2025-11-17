# GPU Performance Testing - Complete Build & Run Guide

This guide walks you through generating all binaries and CSV files from scratch for the GPU performance benchmarking suite.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Project Structure](#project-structure)
3. [Quick Start (Automated)](#quick-start-automated)
4. [Step-by-Step Manual Process](#step-by-step-manual-process)
5. [Understanding the Pipeline](#understanding-the-pipeline)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

- **NVIDIA GPU**: 2080 Ti (or modify for your card)
- **CUDA Toolkit**: Version 10.0+ with `nvcc` compiler
- **Python 3**: Version 3.6+ with standard library
- **Bash**: For running shell scripts
- **nvidia-smi**: NVIDIA System Management Interface

### Verify Installation

```bash
# Check CUDA compiler
nvcc --version

# Check GPU
nvidia-smi

# Check Python
python3 --version
```

---

## Project Structure

```
gpu-perf/
├── bin/                          # Compiled binaries (generated)
│   ├── runner                    # Main benchmark runner
│   ├── props                     # GPU properties collector
│   ├── stream_like               # Memory bandwidth calibration
│   └── gemm_cublas               # Compute throughput calibration
│
├── calibration/                  # GPU calibration tools (source)
│   ├── props.cu                  # Queries GPU properties
│   ├── stream_like.cu            # Measures peak memory bandwidth
│   └── gemm_cublas.cu            # Measures peak compute throughput
│
├── kernels/                      # CUDA kernel implementations
│   ├── vector_add.cuh
│   ├── matmul_naive.cuh
│   ├── matmul_tiled.cuh
│   └── ... (16 total kernels)
│
├── runner/                       # Benchmark runner (source)
│   └── main.cu                   # Orchestrates kernel execution
│
├── scripts/                      # Python/Bash automation scripts
│   ├── run_2080ti_full.sh       # FULL PIPELINE (recommended)
│   ├── gen_trials_2080ti.sh     # Generate trials only
│   ├── run_trials.sh            # Run single kernel trials
│   ├── aggregate_trials.py      # Aggregate trial CSVs
│   ├── make_final_2080ti.py     # Generate final enriched CSV
│   └── fix_block_column.py      # Fix corrupted block columns
│
└── data/                         # Generated data (output)
    ├── props_2080ti.out          # GPU properties
    ├── stream_like_2080ti.out    # Memory bandwidth results
    ├── gemm_cublas_2080ti.out    # Compute throughput results
    ├── trials_*__2080ti.csv      # Per-kernel trial data (16 files)
    └── runs_2080ti_final.csv     # Final enriched dataset
```

---

## Quick Start (Automated)

### Option 1: Full Pipeline (Recommended)

This script does **everything** from scratch:

```bash
cd gpu-perf
chmod +x scripts/run_2080ti_full.sh
./scripts/run_2080ti_full.sh
```

**What it does:**
1. Cleans old binaries and CSVs
2. Builds calibration tools
3. Runs GPU calibration (props, bandwidth, compute)
4. Builds benchmark runner
5. Runs 10 trials for each of 16 kernels
6. Aggregates and enriches data
7. Generates final CSV: `data/runs_2080ti_final.csv`

**Duration:** ~10-15 minutes (depending on GPU and warmup/reps)

### Option 2: Simplified Pipeline (Using Updated Script)

If you want to use the streamlined approach with `make_final_2080ti.py`:

```bash
cd gpu-perf

# 1. Build binaries
nvcc -O3 -arch=sm_75 -o bin/props calibration/props.cu
nvcc -O3 -arch=sm_75 -o bin/stream_like calibration/stream_like.cu
nvcc -O3 -arch=sm_75 -lcublas -o bin/gemm_cublas calibration/gemm_cublas.cu

# 2. Run calibration
bin/props > data/props_2080ti.out
bin/stream_like > data/stream_like_2080ti.out
bin/gemm_cublas > data/gemm_cublas_2080ti.out

# 3. Build runner
nvcc -std=c++14 -O3 --ptxas-options=-v -lineinfo -arch=sm_75 -DTILE=32 \
  -o bin/runner runner/main.cu 2> data/ptxas_2080ti.log

# 4. Generate trials (run all kernels)
chmod +x scripts/gen_trials_2080ti.sh
./scripts/gen_trials_2080ti.sh

# 5. Fix any corrupted block columns (if needed)
python3 scripts/fix_block_column.py data/trials_*__2080ti.csv

# 6. Generate final CSV with all metrics
python3 scripts/make_final_2080ti.py
```

**Output:** `data/runs_2080ti_final.csv` with clean schema (no unused columns)

---

## Step-by-Step Manual Process

### Step 1: Build Calibration Tools

```bash
cd gpu-perf
mkdir -p bin data

# Compile GPU properties tool
nvcc -O3 -arch=sm_75 -o bin/props calibration/props.cu

# Compile memory bandwidth benchmark
nvcc -O3 -arch=sm_75 -o bin/stream_like calibration/stream_like.cu

# Compile compute throughput benchmark (requires cuBLAS)
nvcc -O3 -arch=sm_75 -lcublas -o bin/gemm_cublas calibration/gemm_cublas.cu
```

**Note:** Adjust `-arch=sm_75` for your GPU:
- **sm_75**: Turing (RTX 2080 Ti, 2080, etc.)
- **sm_86**: Ampere (RTX 3090, 3080, etc.)
- **sm_89**: Ada Lovelace (RTX 4090, 4080, etc.)

### Step 2: Run GPU Calibration

```bash
# Collect GPU properties (SM count, memory, etc.)
bin/props > data/props_2080ti.out

# Measure sustained memory bandwidth (GB/s)
bin/stream_like > data/stream_like_2080ti.out

# Measure sustained compute throughput (GFLOPS)
bin/gemm_cublas > data/gemm_cublas_2080ti.out
```

**Inspect results:**
```bash
cat data/props_2080ti.out
cat data/stream_like_2080ti.out | grep "SUSTAINED_MEM_BW_GBPS"
cat data/gemm_cublas_2080ti.out | grep "SUSTAINED_COMPUTE_GFLOPS"
```

### Step 3: Build Benchmark Runner

```bash
# Compile main runner with all kernels
nvcc -std=c++14 -O3 --ptxas-options=-v -lineinfo \
  -arch=sm_75 -DTILE=32 \
  -o bin/runner runner/main.cu \
  2> data/ptxas_2080ti.log

# Verify it works
bin/runner --kernel vector_add --rows 1048576 --cols 1 --block 256 --warmup 5 --reps 10
```

### Step 4: Run Trials for All Kernels

You can run trials manually or use the automation script.

#### Option A: Automated (Recommended)

```bash
chmod +x scripts/run_trials.sh scripts/gen_trials_2080ti.sh
./scripts/gen_trials_2080ti.sh
```

This generates 16 CSV files:
- `data/trials_vector_add__2080ti.csv`
- `data/trials_matmul_naive__2080ti.csv`
- ... (one per kernel)

#### Option B: Manual Single Kernel

```bash
# Run 10 trials of vector_add
scripts/run_trials.sh vector_add \
  "--rows 1048576 --cols 1 --block 256 --warmup 20 --reps 100" \
  12 0 10

# Check output
cat data/trials_vector_add__2080ti.csv
```

**Script parameters:**
- `$1`: kernel name
- `$2`: kernel arguments (problem size, warmup, reps)
- `$3`: register count (annotation)
- `$4`: shared memory bytes (annotation)
- `$5`: number of trials to run

### Step 5: Fix Data Inconsistencies (If Needed)

```bash
# Fix corrupted 'block' column values
python3 scripts/fix_block_column.py data/trials_*__2080ti.csv
```

**What it fixes:**
- Recalculates `block` column (threads per block) from `block_x × block_y × block_z`
- Recalculates `grid_blocks` column from `grid_x × grid_y × grid_z`
- Fixes malformed values like `(16,` → `256`

### Step 6: Generate Final Enriched CSV

```bash
# Generate final CSV with all metrics and cleaned schema
python3 scripts/make_final_2080ti.py
```

**Output:** `data/runs_2080ti_final.csv`

**Columns in final CSV:**
```
kernel, args, device_name, regs, shmem,
block_x, block_y, block_z, grid_x, grid_y, grid_z,
warmup, reps, trials, mean_ms, std_ms,
rows, cols, iters, block, grid_blocks,
FLOPs, BYTES, arithmetic_intensity, working_set_bytes, shared_bytes, mem_pattern,
gpu_name, compute_capability, sm_count, warp_size,
max_threads_per_sm, max_blocks_per_sm, registers_per_sm, shared_mem_per_sm,
peak_theoretical_gflops, peak_theoretical_gbps,
sustained_compute_gflops, sustained_mem_bandwidth_gbps,
gpu_l2_bytes, estimated_l2_bytes,
T1_model_ms, speedup_model
```

**Removed columns** (from previous versions):
- `N`, `matN`, `H`, `W` - Always empty, redundant with `rows`/`cols`

---

## Understanding the Pipeline

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ 1. BUILD PHASE                                              │
├─────────────────────────────────────────────────────────────┤
│ calibration/*.cu  →  nvcc  →  bin/props                     │
│                              →  bin/stream_like             │
│                              →  bin/gemm_cublas             │
│ runner/main.cu    →  nvcc  →  bin/runner                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. CALIBRATION PHASE                                        │
├─────────────────────────────────────────────────────────────┤
│ bin/props        →  data/props_2080ti.out                   │
│ bin/stream_like  →  data/stream_like_2080ti.out             │
│ bin/gemm_cublas  →  data/gemm_cublas_2080ti.out             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. TRIALS PHASE (for each kernel, N=10 trials)             │
├─────────────────────────────────────────────────────────────┤
│ scripts/run_trials.sh  →  bin/runner --kernel vector_add   │
│                        →  data/trials_vector_add__2080ti.csv│
│                        →  ... (repeat for 16 kernels)       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. AGGREGATION & ENRICHMENT PHASE                           │
├─────────────────────────────────────────────────────────────┤
│ scripts/fix_block_column.py  →  Fix corrupted values       │
│                                                              │
│ scripts/make_final_2080ti.py  →  Reads:                    │
│   • data/trials_*__2080ti.csv (all 16 files)               │
│   • data/props_2080ti.out                                   │
│   • data/stream_like_2080ti.out                             │
│   • data/gemm_cublas_2080ti.out                             │
│                                                              │
│   Computes:                                                  │
│   • Aggregated mean/std across trials                       │
│   • Static FLOPs, BYTES, arithmetic intensity               │
│   • GPU metadata (SM count, warp size, etc.)                │
│   • Roofline model (T1 baseline, speedup)                   │
│                                                              │
│   Generates:                                                 │
│   • data/runs_2080ti_final.csv                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Concepts

#### 1. **Trials** (Raw Measurements)
- Each kernel runs N times (default: 10 trials)
- Each trial includes warmup iterations + timed repetitions
- Output: One CSV per kernel with individual trial times

#### 2. **Aggregation** (Statistics)
- Computes `mean_ms` and `std_ms` across trials
- Groups by unique configuration (kernel + args + grid/block dims)
- Output: Single aggregated row per configuration

#### 3. **Static Counts** (FLOPs/BYTES)
- Analytically computes operations and memory traffic per kernel
- Based on problem size (N, rows×cols, etc.)
- Includes arithmetic intensity (FLOPs/BYTES ratio)

#### 4. **GPU Metrics** (Hardware Info)
- SM count, warp size, register/shared memory limits
- Sustained bandwidth and compute throughput
- L2 cache size

#### 5. **Roofline Model** (Performance Bounds)
- `T1_model_ms`: Single-thread baseline time
- `speedup_model`: Actual time / T1 time

---

## Kernel List (16 Total)

| Kernel                   | Problem Size              | Description                          |
|--------------------------|---------------------------|--------------------------------------|
| `vector_add`             | N=1M elements             | Basic element-wise addition          |
| `saxpy`                  | N=1M elements             | `y = a*x + y`                        |
| `strided_copy_8`         | N=1M elements             | Copy with stride-8 access            |
| `naive_transpose`        | 2048×2048 matrix          | Transpose without shared memory      |
| `shared_transpose`       | 2048×2048 matrix          | Transpose using shared memory        |
| `matmul_naive`           | 512×512 matrices          | Matrix multiply (naive)              |
| `matmul_tiled`           | 512×512 matrices          | Matrix multiply (tiled/shared)       |
| `reduce_sum`             | N=1M elements             | Parallel sum reduction               |
| `dot_product`            | N=1M elements             | Dot product with reduction           |
| `histogram`              | N=1M elements             | Atomic histogram binning             |
| `conv2d_3x3`             | 1024×1024 image           | 2D convolution (3×3 kernel)          |
| `conv2d_7x7`             | 1024×1024 image           | 2D convolution (7×7 kernel)          |
| `random_access`          | N=1M elements             | Random gather/scatter                |
| `vector_add_divergent`   | N=1M elements             | Vector add with branch divergence    |
| `shared_bank_conflict`   | Micro-kernel              | Stress-test bank conflicts           |
| `atomic_hotspot`         | N=1M, 100 iters           | Atomic operations on single location |

---

## Troubleshooting

### Issue: `nvcc: command not found`

**Solution:** Install CUDA Toolkit or add to PATH:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Issue: `undefined reference to cublasCreate_v2`

**Solution:** Link cuBLAS explicitly:
```bash
nvcc -O3 -arch=sm_75 -lcublas -o bin/gemm_cublas calibration/gemm_cublas.cu
```

### Issue: Wrong GPU architecture

**Error:** `no kernel image available for execution`

**Solution:** Check your GPU's compute capability:
```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

Then compile with correct `-arch` flag:
- Turing (2080 Ti): `-arch=sm_75`
- Ampere (3090): `-arch=sm_86`
- Ada (4090): `-arch=sm_89`

### Issue: Corrupted `block` column in CSV

**Symptom:** Values like `(16,` instead of `256`

**Solution:** Run the fix script:
```bash
python3 scripts/fix_block_column.py data/trials_*__2080ti.csv
```

### Issue: Empty or missing calibration data

**Symptom:** Final CSV has blank GPU metrics

**Solution:** Re-run calibration:
```bash
bin/props > data/props_2080ti.out
bin/stream_like > data/stream_like_2080ti.out
bin/gemm_cublas > data/gemm_cublas_2080ti.out

# Verify files are not empty
wc -l data/*.out
```

### Issue: Trials fail with timeout

**Solution:** Reduce problem size or warmup/reps in `gen_trials_2080ti.sh`:
```bash
# Change:
scripts/run_trials.sh vector_add "--rows 1048576 --block 256 --warmup 20 --reps 100" ...

# To:
scripts/run_trials.sh vector_add "--rows 1048576 --block 256 --warmup 5 --reps 10" ...
```

---

## Additional Scripts

### Validate CSV Schema
```bash
python3 scripts/validate_csv.py data/runs_2080ti_final.csv
```

### Retrofit Old CSVs to New Schema
```bash
python3 scripts/retrofit_trials_columns.py data/trials_old__2080ti.csv
```

### Verify Calibration
```bash
python3 scripts/verify_calibration.py
```

---

## Quick Reference Commands

```bash
# Clean everything
rm -rf bin data/*.csv data/*.out

# Rebuild binaries only
nvcc -O3 -arch=sm_75 -o bin/props calibration/props.cu
nvcc -O3 -arch=sm_75 -o bin/stream_like calibration/stream_like.cu
nvcc -O3 -arch=sm_75 -lcublas -o bin/gemm_cublas calibration/gemm_cublas.cu
nvcc -std=c++14 -O3 -arch=sm_75 -DTILE=32 -o bin/runner runner/main.cu

# Run single kernel trial
scripts/run_trials.sh vector_add "--rows 1048576 --block 256 --warmup 20 --reps 100" 12 0 10

# Generate final CSV
python3 scripts/make_final_2080ti.py

# Check results
head -5 data/runs_2080ti_final.csv
wc -l data/runs_2080ti_final.csv  # Should be 17 lines (header + 16 kernels)
```

---

## Summary of Recent Fixes

### What Was Fixed (2024-11-17)

1. **Corrupted `block` column** - Fixed values like `(16,` → `256`
2. **Removed unused columns** - Dropped `N`, `matN`, `H`, `W` (always empty)
3. **Updated `make_final_2080ti.py`** - Cleaner schema, correct aggregation keys

### Files Modified
- `scripts/fix_block_column.py` (new)
- `scripts/make_final_2080ti.py` (updated)
- All `data/trials_*__2080ti.csv` (fixed)
- `data/runs_2080ti_final.csv` (regenerated)

---

## Support

For issues or questions, refer to:
- Project README (if exists)
- CUDA documentation: https://docs.nvidia.com/cuda/
- Check Git commit history for recent changes

---

**Generated:** 2024-11-17
**GPU Target:** NVIDIA RTX 2080 Ti (sm_75)
**Last Updated:** After fixing CSV inconsistencies
