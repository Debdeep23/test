#!/usr/bin/env python3
"""
Fix corrupted 'block' column in trials CSV files.
The 'block' column should contain threads_per_block (block_x * block_y * block_z)
but some CSVs have corrupted data like '(16' instead of '256'.

Usage:
  python3 scripts/fix_block_column.py data/trials_*.csv
"""

import csv
import sys
import glob
from pathlib import Path
from tempfile import NamedTemporaryFile
import shutil

def fix_csv(path: Path):
    """Fix the block and grid_blocks columns by recalculating from dimensions."""
    rows_fixed = 0

    with open(path, newline="") as f_in, NamedTemporaryFile("w", newline="", delete=False) as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            # Recalculate block (threads per block)
            try:
                bx = int(row.get('block_x', '1'))
                by = int(row.get('block_y', '1'))
                bz = int(row.get('block_z', '1'))
                threads_per_block = bx * by * bz

                # Recalculate grid_blocks (total blocks)
                gx = int(row.get('grid_x', '1'))
                gy = int(row.get('grid_y', '1'))
                gz = int(row.get('grid_z', '1'))
                total_blocks = gx * gy * gz

                # Check if values need fixing
                old_block = row.get('block', '')
                old_grid = row.get('grid_blocks', '')

                if old_block != str(threads_per_block) or old_grid != str(total_blocks):
                    rows_fixed += 1

                row['block'] = str(threads_per_block)
                row['grid_blocks'] = str(total_blocks)

            except (ValueError, KeyError) as e:
                print(f"Warning: Could not fix row in {path}: {e}", file=sys.stderr)

            writer.writerow(row)

    shutil.move(f_out.name, path)
    return rows_fixed

def main():
    if len(sys.argv) < 2:
        print("Usage: scripts/fix_block_column.py data/trials_*.csv", file=sys.stderr)
        sys.exit(2)

    files = []
    for pattern in sys.argv[1:]:
        files.extend(glob.glob(pattern))

    if not files:
        print("No files matched the pattern", file=sys.stderr)
        sys.exit(1)

    total_fixed = 0
    for f in sorted(files):
        path = Path(f)
        if path.exists():
            fixed = fix_csv(path)
            total_fixed += fixed
            print(f"[OK] Fixed {fixed} rows in {path}")

    print(f"\n[DONE] Fixed {total_fixed} total rows across {len(files)} files")

if __name__ == "__main__":
    main()
