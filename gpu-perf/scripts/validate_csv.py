#!/usr/bin/env python3
"""
Lenient sanity checks for per-trial CSVs.

- Does NOT require a fixed header. It will tolerate missing columns and treat them as zeros.
- Checks time_ms > 0, nonzero block/grid (or warns), and presence of at least ONE size family
  unless the kernel is explicitly sizeless (e.g., shared_bank_conflict).

Usage:
  scripts/validate_csv.py data/trials_<kernel>__<tag>.csv
"""

import csv
import sys

SIZeless = {"shared_bank_conflict"}  # kernels that truly have no problem size

# Columns we try to read; missing ones default to 0 and only emit warnings
SOFT_FIELDS = [
    "kernel","args","regs","shmem","device_name",
    "block_x","block_y","block_z","grid_x","grid_y","grid_z",
    "warmup","reps","trials","time_ms",
    "N","rows","cols","matN","H","W","iters","block","grid_blocks"
]

def I(row, k):
    try:
        return int(row.get(k, "0"))
    except Exception:
        return 0

def F(row, k):
    try:
        return float(row.get(k, "0"))
    except Exception:
        return 0.0

def validate(path: str) -> int:
    issues = []
    warnings = []

    try:
        with open(path, newline="") as f:
            rd = csv.DictReader(f)

            # soft header check: warn but continue
            missing = [c for c in SOFT_FIELDS if c not in (rd.fieldnames or [])]
            if missing:
                warnings.append(f"{path}: missing fields (will default to 0): {', '.join(missing)}")

            for ln, row in enumerate(rd, start=2):
                k = (row.get("kernel") or "").strip()

                bx,by,bz = I(row,"block_x"), I(row,"block_y"), I(row,"block_z")
                gx,gy,gz = I(row,"grid_x"),  I(row,"grid_y"),  I(row,"grid_z")
                t        = F(row,"time_ms")

                if t <= 0:
                    issues.append(f"[{path}:{ln}] {k}: non-positive time_ms={row.get('time_ms')}")

                # block/grid sanity
                if bx <= 0 or by <= 0 or bz <= 0 or gx <= 0 or gy <= 0 or gz <= 0:
                    issues.append(f"[{path}:{ln}] {k}: suspicious block/grid: "
                                  f"block=({bx},{by},{bz}) grid=({gx},{gy},{gz})")

                # size family presence (unless sizeless)
                if k not in SIZeless:
                    N  = I(row,"N")
                    r  = I(row,"rows")
                    c  = I(row,"cols")
                    H  = I(row,"H")
                    W  = I(row,"W")
                    mN = I(row,"matN")
                    if not ((N>0) or (r>0 and c>0) or (H>0 and W>0) or (mN>0)):
                        issues.append(f"[{path}:{ln}] {k}: missing problem size (N/rows&cols/H&W/matN all 0 or missing)")

                # internal consistency if present
                block_flat = I(row,"block")
                grid_blks  = I(row,"grid_blocks")
                if block_flat and block_flat != max(1,bx)*max(1,by)*max(1,bz):
                    warnings.append(f"[{path}:{ln}] {k}: block mismatch flat={block_flat} vs {bx*by*bz}")
                if grid_blks and grid_blks != max(1,gx)*max(1,gy)*max(1,gz):
                    warnings.append(f"[{path}:{ln}] {k}: grid_blocks mismatch {grid_blks} vs {gx*gy*gz}")

    except FileNotFoundError:
        print(f"[FAIL] {path}: not found")
        return 1

    # Print results
    if warnings:
        for w in warnings:
            print(w)
    if issues:
        for msg in issues:
            print(msg)
        print(f"[FAIL] {path} had {len(issues)} issues")
        return 1
    else:
        print(f"[OK] {path} passed basic validation")
        return 0

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: scripts/validate_csv.py data/trials_<kernel>__<tag>.csv", file=sys.stderr)
        sys.exit(2)
    sys.exit(validate(sys.argv[1]))

