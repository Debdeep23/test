# GPU-Perf Quick Start Guide

## 🚀 One-Command Full Pipeline

```bash
cd gpu-perf
chmod +x scripts/run_2080ti_full.sh
./scripts/run_2080ti_full.sh
```

**Output:** `data/runs_2080ti_final.csv` (16 kernels, fully enriched metrics)

---

## 📋 Prerequisites

```bash
# Check you have everything
nvcc --version          # CUDA compiler
nvidia-smi             # GPU visible
python3 --version      # Python 3.6+
```

---

## 🔧 Step-by-Step Alternative

### 1. Build Everything

```bash
# Calibration tools
nvcc -O3 -arch=sm_75 -o bin/props calibration/props.cu
nvcc -O3 -arch=sm_75 -o bin/stream_like calibration/stream_like.cu
nvcc -O3 -arch=sm_75 -lcublas -o bin/gemm_cublas calibration/gemm_cublas.cu

# Runner
nvcc -std=c++14 -O3 -arch=sm_75 -DTILE=32 -o bin/runner runner/main.cu
```

### 2. Calibrate GPU

```bash
bin/props > data/props_2080ti.out
bin/stream_like > data/stream_like_2080ti.out
bin/gemm_cublas > data/gemm_cublas_2080ti.out
```

### 3. Run Trials

```bash
chmod +x scripts/gen_trials_2080ti.sh
./scripts/gen_trials_2080ti.sh
```

This creates 16 CSV files in `data/trials_*__2080ti.csv`

### 4. Generate Final CSV

```bash
# Fix any data issues first
python3 scripts/fix_block_column.py data/trials_*__2080ti.csv

# Generate enriched CSV
python3 scripts/make_final_2080ti.py
```

**Result:** `data/runs_2080ti_final.csv`

---

## 📊 What You Get

### Per-Kernel Trial Files
- `data/trials_vector_add__2080ti.csv`
- `data/trials_matmul_naive__2080ti.csv`
- ... (16 total)

Each contains 10 timing trials with grid/block dimensions.

### Final Enriched CSV
`data/runs_2080ti_final.csv` with:
- Mean/std across trials
- FLOPs, BYTES, arithmetic intensity
- GPU metrics (SM count, bandwidth, compute throughput)
- Roofline model (baseline time, speedup)

---

## 🎯 Run Single Kernel

```bash
# Example: vector_add with 10 trials
scripts/run_trials.sh vector_add \
  "--rows 1048576 --cols 1 --block 256 --warmup 20 --reps 100" \
  12 0 10
```

**Parameters:**
1. Kernel name
2. Kernel args (problem size + timing params)
3. Register count (annotation)
4. Shared memory bytes (annotation)
5. Number of trials

---

## 🐛 Common Issues

### Problem: `nvcc: command not found`
```bash
export PATH=/usr/local/cuda/bin:$PATH
```

### Problem: Wrong architecture
Check your GPU:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```

Update `-arch` flag:
- `sm_75` = Turing (2080 Ti)
- `sm_86` = Ampere (3090)
- `sm_89` = Ada (4090)

### Problem: Corrupted CSV data
```bash
python3 scripts/fix_block_column.py data/trials_*__2080ti.csv
```

---

## 📁 Key Files

| Path | Description |
|------|-------------|
| `bin/runner` | Main benchmark binary |
| `data/runs_2080ti_final.csv` | **Final output** |
| `scripts/run_2080ti_full.sh` | Complete pipeline script |
| `scripts/make_final_2080ti.py` | Final CSV generator |
| `BUILD_AND_RUN_GUIDE.md` | Full documentation |

---

## ⏱️ Expected Duration

- Full pipeline: **10-15 minutes**
- Single kernel: **~30 seconds**

---

## 🔍 Verify Results

```bash
# Check final CSV
head -3 data/runs_2080ti_final.csv
wc -l data/runs_2080ti_final.csv   # Should be 17 (header + 16 kernels)

# Check a trial file
head -3 data/trials_vector_add__2080ti.csv
```

---

## 📚 More Info

See `BUILD_AND_RUN_GUIDE.md` for:
- Detailed pipeline explanation
- Kernel descriptions
- Troubleshooting guide
- CSV schema documentation

---

**Updated:** 2024-11-17 (after fixing CSV inconsistencies)
