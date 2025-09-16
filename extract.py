# extract.py
import re
import math
import warnings
import argparse
from pathlib import Path
from typing import Tuple, Optional

from PIL import Image, ImageFile

# ---------- SETTINGS ----------
IN_DIR = Path("input_quilts")
OUT_DIR = Path("output_center_views")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PIL_JPEG_QUALITY = 98
PIL_JPEG_SUBSAMPLING = 0
# ------------------------------

# Disable PIL "decompression bomb" safeguards for giant but valid quilts.
Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter("ignore", Image.DecompressionBombWarning)
ImageFile.LOAD_TRUNCATED_IMAGES = True

_QS_RE = re.compile(r"qs(?P<cols>\d+)x(?P<rows>\d+)a(?P<a>[0-9]*\.?[0-9]+)", re.IGNORECASE)

def parse_quilt_spec_from_name(name: str) -> Tuple[int, int, float, str]:
    m = _QS_RE.search(name)
    if not m:
        raise ValueError(f"Could not find 'qs{{C}}x{{R}}a{{A}}' in filename: {name}")
    cols = int(m.group("cols"))
    rows = int(m.group("rows"))
    a_str = m.group("a")
    a_val = float(a_str)
    if cols <= 0 or rows <= 0:
        raise ValueError(f"Non-positive grid parsed from '{name}': cols={cols}, rows={rows}")
    return cols, rows, a_val, a_str

def center_view_index(cols: int, rows: int) -> int:
    total = cols * rows
    if total <= 0:
        return 0
    return int(math.floor((total - 1) / 2.0))

def tile_box_for_index(img_w: int, img_h: int, cols: int, rows: int, index: int) -> Tuple[int, int, int, int]:
    if cols <= 0 or rows <= 0:
        raise ValueError("cols and rows must be positive")
    r = index // cols
    c = index % cols
    if r < 0 or r >= rows or c < 0 or c >= cols:
        raise ValueError(f"Index {index} out of range for grid {cols}x{rows}")
    rr = (rows - 1) - r
    tile_w_f = img_w / float(cols)
    tile_h_f = img_h / float(rows)
    x0 = int(round(c * tile_w_f))
    x1 = int(round((c + 1) * tile_w_f))
    y0 = int(round(rr * tile_h_f))
    y1 = int(round((rr + 1) * tile_h_f))
    x0 = max(0, min(x0, img_w))
    x1 = max(0, min(x1, img_w))
    y0 = max(0, min(y0, img_h))
    y1 = max(0, min(y1, img_h))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Degenerate tile box for index {index} in {cols}x{rows} grid on {img_w}x{img_h}")
    return x0, y0, x1, y1

def stretch_to_aspect(img: Image.Image, target_aspect: float) -> Image.Image:
    if target_aspect <= 0:
        return img
    w, h = img.size
    if w <= 0 or h <= 0:
        return img
    cur_aspect = w / float(h)
    if abs(cur_aspect - target_aspect) < 1e-6:
        return img
    if cur_aspect > target_aspect:
        new_w = int(round(h * target_aspect))
        new_h = h
    else:
        new_w = w
        new_h = int(round(w / target_aspect))
    new_w = max(1, new_w)
    new_h = max(1, new_h)
    return img.resize((new_w, new_h), resample=Image.BICUBIC)

def extract_center_view_from_quilt(src_path: Path, out_dir: Path) -> Optional[Path]:
    try:
        cols, rows, a_val, a_str = parse_quilt_spec_from_name(src_path.stem)
    except Exception as e:
        print(f"[WARN] {src_path.name}: {e}")
        return None
    try:
        with Image.open(src_path) as im:
            im = im.convert("RGB")
            img_w, img_h = im.size
            idx = center_view_index(cols, rows)
            x0, y0, x1, y1 = tile_box_for_index(img_w, img_h, cols, rows, idx)
            tile = im.crop((x0, y0, x1, y1))
            tile_fixed = stretch_to_aspect(tile, a_val)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_name = f"{src_path.stem}_center.jpg"
            out_path = out_dir / out_name
            tile_fixed.save(str(out_path), format="JPEG", quality=PIL_JPEG_QUALITY, subsampling=PIL_JPEG_SUBSAMPLING, optimize=False, progressive=False)
            print(f"Saved center view: {out_path.name}  [grid {cols}x{rows}, a={a_str}]")
            return out_path
    except Exception as e:
        print(f"[ERROR] {src_path.name}: {e}")
        return None

def is_quilt_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def main():
    ap = argparse.ArgumentParser(description="Extract the center view from quilt images by parsing qs{cols}x{rows}a{aspect} from the filename, then stretch-resizing the tile to the specified aspect ratio (no cropping).")
    ap.add_argument("--in_dir", type=str, default=str(IN_DIR), help="Directory containing quilt images (filenames include qs{C}x{R}a{A}).")
    ap.add_argument("--out_dir", type=str, default=str(OUT_DIR), help="Directory to write extracted center-view JPEGs.")
    args = ap.parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = [p for p in in_dir.iterdir() if is_quilt_image(p)]
    if not imgs:
        print(f"No quilt images found in {in_dir}.")
        return
    print("Selected quilt images:")
    for p in imgs:
        print("  -", p.name)
    ok_count = 0
    for p in imgs:
        out = extract_center_view_from_quilt(p, out_dir)
        if out is not None:
            ok_count += 1
    print(f"Done. Extracted {ok_count}/{len(imgs)} center view(s).")

if __name__ == "__main__":
    main()
