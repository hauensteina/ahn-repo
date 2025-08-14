#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import os

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

# Enable HEIC/HEIF support in Pillow
register_heif_opener()

#--------------
def main():
    ap = argparse.ArgumentParser(description="Convert HEIC images to JPEG.")
    ap.add_argument("input_dir", type=Path, help="Folder containing HEIC files")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="Output folder (default: create JPEGs next to sources)")
    ap.add_argument("-r", "--recursive", action="store_true",
                    help="Recurse into subfolders")
    ap.add_argument("-q", "--quality", type=int, default=92,
                    help="JPEG quality (1–95, default 92)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing JPEGs if present")
    ap.add_argument("--keep-times", action="store_true",
                    help="Preserve file modified/ accessed times")
    args = ap.parse_args()

    base = args.input_dir.resolve()
    if not base.exists() or not base.is_dir():
        print(f"Error: {base} is not a directory", file=sys.stderr)
        sys.exit(1)

    files = find_heic_files(base, args.recursive)
    if not files:
        print("No .heic files found.")
        return

    for src in files:
        if args.output:
            # Mirror directory structure under output
            rel = src.relative_to(base)
            dst = (args.output / rel).with_suffix(".jpg")
        else:
            dst = src.with_suffix(".jpg")

        try:
            convert_one(src, dst, args.quality, args.overwrite, args.keep_times)
        except Exception as e:
            print(f"FAIL: {src} -> {dst} ({e})", file=sys.stderr)

#---------------------------------------------------
def find_heic_files(base: Path, recursive: bool):
    if recursive:
        return [p for p in base.rglob("*") if p.is_file() and p.suffix.lower() == ".heic"]
    else:
        return [p for p in base.iterdir() if p.is_file() and p.suffix.lower() == ".heic"]

#----------------------------------------------------------------------------------------
def convert_one(src: Path, dst: Path, quality: int, overwrite: bool, keep_times: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        print(f"SKIP (exists): {dst}")
        return

    with Image.open(src) as im:
        # Apply correct orientation if EXIF has rotation
        im = ImageOps.exif_transpose(im)

        exif = im.info.get("exif")
        icc = im.info.get("icc_profile")

        save_kwargs = {
            "format": "JPEG",
            "quality": quality,
            "optimize": True,
            "progressive": True,
            #"subsampling": "keep",   # keep or auto: preserves 4:4:4 if present
        }
        if exif:
            save_kwargs["exif"] = exif
        if icc:
            save_kwargs["icc_profile"] = icc

        im.convert("RGB").save(dst, **save_kwargs)

    if keep_times:
        st = src.stat()
        os.utime(dst, (st.st_atime, st.st_mtime))

    print(f"OK: {src} -> {dst}")


if __name__ == "__main__":
    main()
