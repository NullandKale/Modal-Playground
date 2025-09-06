# fix_and_quilt.py
import os
import re
import json
import math
import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional

# ---------- SETTINGS ----------
DEFAULT_COLS = 10
DEFAULT_ROWS = 10
FINAL_OUT_W = 8192
FINAL_OUT_H = 8192
# ------------------------------

IN_DIR = Path("output")
OUT_DIR = Path("output_quilts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_EXTS = {".mp4", ".mov", ".webm", ".m4v"}
_INTERMEDIATE_SUFFIX_RE = re.compile(r"[\s_\-](left|right|rail|input)$", re.IGNORECASE)

# ---- FFmpeg quality knobs (centralized) ----
# Lower CRF = higher quality. 12 is a good near-lossless mezzanine for generated content.
H264_CRF = 12
H264_PRESET = "slow"
H264_PIXFMT = "yuv420p"  # widely compatible; keeps inputs consistent for concat
JPEG_QUALITY = "1"       # best quality for libjpeg in ffmpeg ("-q:v 1")

def _h264_args() -> list[str]:
    return [
        "-c:v", "libx264",
        "-crf", str(H264_CRF),
        "-preset", H264_PRESET,
        "-pix_fmt", H264_PIXFMT,
        "-movflags", "+faststart",
    ]
# --------------------------------------------

def run(cmd: list[str]) -> None:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line.rstrip())
        log.append(line)
    rc = proc.wait()
    if rc != 0:
        tail = "".join(log).splitlines()[-120:]
        raise RuntimeError("Command failed:\n" + "\n".join(tail))

def is_video(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTS

def is_final_output_video(path: Path) -> bool:
    # excludes *_left/_right/_rail/_input
    return _INTERMEDIATE_SUFFIX_RE.search(path.stem) is None

def ffprobe_json(path: Path) -> dict:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
        text=True
    )
    return json.loads(out)

def get_video_wh_and_frames(path: Path) -> Tuple[int, int, Optional[int]]:
    data = ffprobe_json(path)
    vstreams = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
    if not vstreams:
        raise ValueError(f"No video stream in {path}")
    vs = vstreams[0]
    w = int(vs["width"])
    h = int(vs["height"])
    nb = None
    if "nb_frames" in vs and str(vs["nb_frames"]).isdigit():
        nb = int(vs["nb_frames"])
    else:
        dur = None
        try:
            dur = float(vs.get("duration", "nan"))
        except:
            pass
        if dur is None or math.isnan(dur):
            try:
                dur = float(data.get("format", {}).get("duration", "0"))
            except:
                dur = 0.0
        afr = vs.get("avg_frame_rate", "0/1")
        try:
            num, den = afr.split("/")
            afr_val = float(num) / float(den) if float(den) != 0 else 0.0
        except:
            afr_val = 0.0
        if afr_val > 0 and dur > 0:
            nb = int(math.floor(dur * afr_val))
    return w, h, nb

def format_aspect(a: float) -> str:
    s = f"{a:.4f}".rstrip("0").rstrip(".")
    return s if s else "0"

def compute_center_window(n_total: Optional[int], win: int) -> tuple[int, int]:
    if n_total is None:
        start = 0
    else:
        start = max(0, (n_total - win) // 2)
    end = start + win - 1
    return start, end

def repair_video(src: Path, dst: Path) -> None:
    """Re-encode to concat-friendly baseline while minimizing quality loss."""
    # Keep dimensions even & use high-quality H.264 mezzanine settings.
    vf = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    run([
        "ffmpeg", "-y",
        "-i", str(src),
        "-an",
        "-vf", vf,
        *_h264_args(),
        "-map_metadata", "-1",
        str(dst)
    ])

def find_lr_pairs(in_dir: Path) -> List[Tuple[Path, Path, str]]:
    files = [p for p in in_dir.iterdir() if is_video(p)]
    lefts = {}
    rights = {}
    for p in files:
        s = p.stem
        if s.lower().endswith("_left"):
            base = s[:-5]
            lefts[base] = p
        elif s.lower().endswith("_right"):
            base = s[:-6]
            rights[base] = p
    pairs = []
    for base in sorted(set(lefts) & set(rights)):
        pairs.append((lefts[base], rights[base], base))
    return pairs

def merge_left_right(left: Path, right: Path, out_path: Path, repair: bool) -> bool:
    tmp_dir = out_path.parent / ".tmp_merge"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        L = left
        R = right
        if repair:
            Lf = tmp_dir / (left.stem + "_fix.mp4")
            Rf = tmp_dir / (right.stem + "_fix.mp4")
            repair_video(left, Lf)
            repair_video(right, Rf)
            L, R = Lf, Rf

        # Reverse LEFT so playback runs left-most -> center.
        left_rev = tmp_dir / (left.stem + "_rev.mp4")
        run([
            "ffmpeg", "-y",
            "-i", str(L),
            "-vf", "reverse",
            "-an",
            *_h264_args(),
            str(left_rev)
        ])

        # Concatenate left_rev + right (video only), again at high quality.
        run([
            "ffmpeg", "-y",
            "-i", str(left_rev), "-i", str(R),
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
            "-map", "[v]",
            *_h264_args(),
            str(out_path)
        ])
        return True
    except Exception as e:
        print(f"[WARN] merge failed for {left} + {right}: {e}")
        return False
    finally:
        try:
            for p in tmp_dir.iterdir():
                p.unlink(missing_ok=True)
            tmp_dir.rmdir()
        except Exception:
            pass

def build_filter(cols: int, rows: int,
                 center_start: int, center_end: int,
                 step: int,
                 invert_views: bool,
                 out_w: Optional[int], out_h: Optional[int]) -> str:
    """
    1) Trim to centered window.
    2) Optionally decimate by `step` (skip=0->step=1 keeps all).
    3) (Optional) invert 1-D view order (off by default).
    4) vflip, tile=COLSxROWS, vflip => BL->TR quilt indexing.
    """
    chain = [f"trim=start_frame={center_start}:end_frame={center_end+1}"]
    if step > 1:
        chain.append(f"select=not(mod(n\\,{step})),setpts=N/FRAME_RATE/TB")
    if invert_views:
        chain.append("reverse")
    chain.append("vflip")
    chain.append(f"tile={cols}x{rows}")
    chain.append("vflip")
    if out_w and out_h:
        chain.append(f"scale={out_w}:{out_h}:flags=lanczos")
    return ",".join(chain)

def main():
    ap = argparse.ArgumentParser(
        description="1) Merge *_left/_right into <base>.mp4 (high-quality), then 2) make quilt JPEGs from final videos."
    )
    ap.add_argument("--cols", type=int, default=DEFAULT_COLS)
    ap.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    ap.add_argument("--skip", type=int, default=1,
                    help="0: use N center frames; 1: use 2N from center & take every other; etc.")
    ap.add_argument("--invert_views", type=int, default=0,
                    help="0: keep 1-D view order; 1: reverse the 1-D view order.")
    ap.add_argument("--in_dir", type=str, default=str(IN_DIR))
    ap.add_argument("--out_dir", type=str, default=str(OUT_DIR))
    ap.add_argument("--repair_before_merge", type=int, default=1,
                    help="Re-encode left/right to concat-friendly format before merging (default 1).")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cols, rows = args.cols, args.rows
    view_total = cols * rows
    step = max(1, args.skip + 1)
    invert_views = bool(args.invert_views)
    do_repair = bool(args.repair_before_merge)

    # --- Step 1: merge any left/right pairs into <base>.mp4 in-place ---
    pairs = find_lr_pairs(in_dir)
    if pairs:
        print(f"Found {len(pairs)} left/right pair(s) to merge.")
    for left, right, base in pairs:
        merged = in_dir / f"{base}.mp4"
        if merged.exists() and merged.stat().st_size > 0:
            print(f"  - {merged.name} already exists, skipping merge.")
            continue
        ok = merge_left_right(left, right, merged, repair=do_repair)
        if ok:
            print(f"  - merged -> {merged.name}")
        else:
            print(f"  - merge FAILED for base='{base}'")

    # --- Step 2: build quilts only from final videos (no *_left/_right/_rail/_input) ---
    all_vids = [p for p in in_dir.iterdir() if is_video(p)]
    finals = [p for p in all_vids if is_final_output_video(p)]
    if not finals:
        print(f"No final videos found in {in_dir}.")
        return

    print("Selected final videos:")
    for p in finals:
        print("  -", p.name)

    for src in finals:
        try:
            print(f"\n=== Processing: {src.name} ===")
            w, h, nb = get_video_wh_and_frames(src)
            aspect = w / float(h) if h else 1.0
            a_str = format_aspect(aspect)

            window_size = view_total * step
            start_idx, end_idx = compute_center_window(nb, window_size)
            if nb is not None and nb < window_size:
                print(f"  Warning: {src.name} has only {nb} frames; "
                      f"center window wants {window_size}. ffmpeg will trim to available frames.")

            out_name = f"{src.stem}_qs{cols}x{rows}a{a_str}.jpg"
            out_path = out_dir / out_name

            vf = build_filter(cols, rows, start_idx, end_idx, step, invert_views, FINAL_OUT_W, FINAL_OUT_H)
            run([
                "ffmpeg", "-y",
                "-i", str(src),
                "-filter_complex", vf,
                "-frames:v", "1",
                "-q:v", JPEG_QUALITY,
                str(out_path)
            ])
            print(f"Saved quilt: {out_path}")
        except Exception as e:
            print(f"ERROR on {src.name}: {e}")

if __name__ == "__main__":
    main()
