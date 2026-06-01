#!/usr/bin/env python3
"""
Crop all JPGs in a folder to 600x600 by removing the rightmost columns
that exceed a width of 600 pixels.

Images are expected to be 600x600, 601x600, or 602x600. Only those wider
than 600px are modified; everything else is left untouched. Cropping keeps
the leftmost 600 columns (i.e. the excess on the right is discarded).

Usage:
    python crop_to_600.py /path/to/folder
    python crop_to_600.py /path/to/folder --dry-run
"""

import argparse
import sys
from pathlib import Path

from PIL import Image

TARGET_W, TARGET_H = 600, 600
JPG_SUFFIXES = {".jpg", ".jpeg"}


def process_folder(folder: Path, dry_run: bool = False) -> None:
    files = [p for p in folder.iterdir()
             if p.is_file() and p.suffix.lower() in JPG_SUFFIXES]

    if not files:
        print(f"No JPG files found in {folder}")
        return

    print(f"Found {len(files)} JPG file(s) in {folder}")
    if dry_run:
        print("DRY RUN — no files will be modified\n")

    cropped = skipped = errors = 0

    for path in files:
        try:
            with Image.open(path) as im:
                w, h = im.size

                if w == TARGET_W and h == TARGET_H:
                    skipped += 1
                    continue

                if h != TARGET_H:
                    # Unexpected height — leave it alone and warn.
                    print(f"  ! {path.name}: unexpected size {w}x{h}, skipping")
                    skipped += 1
                    continue

                # Crop keeping leftmost TARGET_W columns: (left, upper, right, lower)
                cropped_im = im.crop((0, 0, TARGET_W, TARGET_H))

                if not dry_run:
                    # Preserve JPEG quality reasonably; keep original metadata-free save.
                    cropped_im.save(path, "JPEG", quality=95)

                print(f"  {'(would crop)' if dry_run else 'cropped'} "
                      f"{path.name}: {w}x{h} -> {TARGET_W}x{TARGET_H}")
                cropped += 1

        except Exception as e:
            print(f"  ! {path.name}: ERROR {e}")
            errors += 1

    print(f"\nDone. Cropped: {cropped}, already-correct/skipped: {skipped}, "
          f"errors: {errors}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Crop JPGs to 600x600 by trimming the rightmost excess columns.")
    parser.add_argument("folder", type=Path, help="Folder containing the JPG files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without writing any files")
    args = parser.parse_args()

    if not args.folder.is_dir():
        print(f"Error: {args.folder} is not a directory")
        return 1

    process_folder(args.folder, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
