# fix_and_quilt.py
import os
import re
import json
import math
import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional

from multiprocessing import Pool, cpu_count
import numpy as np
from PIL import Image

# ---------- SETTINGS ----------
DEFAULT_COLS = 10
DEFAULT_ROWS = 10
FINAL_OUT_W = 8192
FINAL_OUT_H = 8192
DEFAULT_PRE_FOCUS = 0.10  # percentage in [-1,1]; -1 = max left, +1 = max right; 0 disables
# ------------------------------

IN_DIR = Path("output")
OUT_DIR = Path("output_quilts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_EXTS = {".mp4", ".mov", ".webm", ".m4v"}
_INTERMEDIATE_SUFFIX_RE = re.compile(r"[\s_\-](left|right|rail|input)$", re.IGNORECASE)

# ---- FFmpeg quality knobs (centralized) ----
H264_CRF = 12
H264_PRESET = "slow"
H264_PIXFMT = "yuv420p"
JPEG_QUALITY = "1"
PIL_JPEG_QUALITY = 98
PIL_JPEG_SUBSAMPLING = 0
# --------------------------------------------

def _h264_args() -> list[str]:
    return [
        "-c:v", "libx264",
        "-crf", str(H264_CRF),
        "-preset", H264_PRESET,
        "-pix_fmt", H264_PIXFMT,
        "-movflags", "+faststart",
    ]

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
    tmp_dir = out_path.parent / f".tmp_merge_{out_path.stem}"
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
        left_rev = tmp_dir / (left.stem + "_rev.mp4")
        run([
            "ffmpeg", "-y",
            "-i", str(L),
            "-vf", "reverse",
            "-an",
            *_h264_args(),
            str(left_rev)
        ])
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

def build_extract_filter(center_start: int, center_end: int, step: int, invert_views: bool) -> str:
    chain = [f"trim=start_frame={center_start}:end_frame={center_end+1}"]
    if step > 1:
        chain.append(f"select=not(mod(n\\,{step})),setpts=N/FRAME_RATE/TB")
    if invert_views:
        chain.append("reverse")
    chain.append("scale=trunc(iw/2)*2:trunc(ih/2)*2")
    return ",".join(chain)

def shift_horiz_edge(img: np.ndarray, shift_px: int) -> np.ndarray:
    if shift_px == 0:
        return img
    h, w, c = img.shape
    if shift_px > 0:
        pad = np.repeat(img[:, :1, :], shift_px, axis=1)
        out = np.concatenate([pad, img[:, :w - shift_px, :]], axis=1)
    else:
        s = -shift_px
        pad = np.repeat(img[:, -1:, :], s, axis=1)
        out = np.concatenate([img[:, s:, :], pad], axis=1)
    return out

def compose_quilt_streaming(src: Path, cols: int, rows: int, start_idx: int, end_idx: int, step: int, invert_views: bool, pre_focus: float, expect_w: int, expect_h: int, final_out_w: int, final_out_h: int, a_str: str, out_dir: Path) -> Path:
    total = cols * rows
    quilt_h = rows * expect_h
    quilt_w = cols * expect_w
    canvas = np.zeros((quilt_h, quilt_w, 3), dtype=np.uint8)
    vf = build_extract_filter(start_idx, end_idx, step, invert_views)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-vf", vf,
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-"
    ]
    bytes_per_frame = expect_w * expect_h * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    vi = 0
    last_frame = None
    last_frame_unshifted = None
    frames_dir = out_dir / f"{src.stem}_qs{cols}x{rows}a{a_str}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    digits = max(1, len(str(total - 1))) if total > 0 else 1
    try:
        while vi < total:
            buf = proc.stdout.read(bytes_per_frame)
            if not buf or len(buf) < bytes_per_frame:
                break
            frame_unshifted = np.frombuffer(buf, dtype=np.uint8).reshape((expect_h, expect_w, 3)).copy()
            img_to_save = Image.fromarray(frame_unshifted, mode="RGB")
            fname = f"{src.stem}_view{vi:0{digits}d}.png"
            img_to_save.save(str(frames_dir / fname), format="PNG", optimize=False)
            frame = frame_unshifted
            if pre_focus != 0.0:
                t = vi / (total - 1) if total > 1 else 0.5
                view_dir = t * 2.0 - 1.0
                shift_px = int(round(view_dir * pre_focus * expect_w))
                frame = shift_horiz_edge(frame_unshifted, shift_px)
            r = vi // cols
            c = vi % cols
            rr = (rows - 1) - r
            y0 = rr * expect_h
            x0 = c * expect_w
            canvas[y0:y0 + expect_h, x0:x0 + expect_w, :] = frame
            last_frame = frame
            last_frame_unshifted = frame_unshifted
            vi += 1
        while vi < total and last_frame is not None:
            if last_frame_unshifted is not None:
                img_to_save = Image.fromarray(last_frame_unshifted, mode="RGB")
                fname = f"{src.stem}_view{vi:0{digits}d}.png"
                img_to_save.save(str(frames_dir / fname), format="PNG", optimize=False)
            frame_use = last_frame
            if pre_focus != 0.0 and last_frame_unshifted is not None:
                t = vi / (total - 1) if total > 1 else 0.5
                view_dir = t * 2.0 - 1.0
                shift_px = int(round(view_dir * pre_focus * expect_w))
                frame_use = shift_horiz_edge(last_frame_unshifted, shift_px)
            r = vi // cols
            c = vi % cols
            rr = (rows - 1) - r
            y0 = rr * expect_h
            x0 = c * expect_w
            canvas[y0:y0 + expect_h, x0:x0 + expect_w, :] = frame_use
            vi += 1
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.wait()
    img = Image.fromarray(canvas, mode="RGB")
    if final_out_w and final_out_h:
        if img.size != (final_out_w, final_out_h):
            img = img.resize((final_out_w, final_out_h), resample=Image.LANCZOS)
    out_name = f"{src.stem}_qs{cols}x{rows}a{a_str}.jpg"
    out_path = out_dir / out_name
    img.save(str(out_path), format="JPEG", quality=PIL_JPEG_QUALITY, subsampling=PIL_JPEG_SUBSAMPLING, optimize=False, progressive=False)
    return out_path

def merge_pair_task(args: tuple) -> tuple[bool, str]:
    left_str, right_str, merged_str, do_repair = args
    left = Path(left_str)
    right = Path(right_str)
    merged = Path(merged_str)
    try:
        if merged.exists() and merged.stat().st_size > 0:
            return True, f"skip merge (exists): {merged.name}"
        ok = merge_left_right(left, right, merged, repair=bool(do_repair))
        return (ok, f"merged -> {merged.name}" if ok else f"merge FAILED for base='{merged.stem}'")
    except Exception as e:
        return False, f"merge exception for {merged.name}: {e}"

def quilt_task(args: tuple) -> tuple[bool, str]:
    src_str, cols, rows, step, invert_views, pre_focus, final_out_w, final_out_h, out_dir_str = args
    try:
        src = Path(src_str)
        out_dir = Path(out_dir_str)
        w, h, nb = get_video_wh_and_frames(src)
        w_even = w - (w % 2)
        h_even = h - (h % 2)
        aspect = w / float(h) if h else 1.0
        a_str = format_aspect(aspect)
        view_total = cols * rows
        window_size = view_total * step
        start_idx, end_idx = compute_center_window(nb, window_size)
        if nb is not None and nb < window_size:
            print(f"[{src.name}] Warning: only {nb} frames; need {window_size}. Will fill by repeating last frame.")
        pf = max(-1.0, min(1.0, float(pre_focus)))
        out_path = compose_quilt_streaming(src, cols, rows, start_idx, end_idx, step, bool(invert_views), pf, w_even, h_even, int(final_out_w), int(final_out_h), a_str, out_dir)
        frames_dir = out_dir / f"{src.stem}_qs{cols}x{rows}a{a_str}_frames"
        return True, f"Saved quilt: {out_path.name} | Frames: {frames_dir.name}"
    except Exception as e:
        return False, f"ERROR on {Path(src_str).name}: {e}"

def main():
    ap = argparse.ArgumentParser(description="1) Merge *_left/_right into <base>.mp4 (high-quality), then 2) make quilt JPEGs from final videos (PIL/NumPy), with optional pre-applied focus percentage [-1..1]. Also saves the selected full-resolution frames into a per-video folder named with the 1-D view index. Processing is split one-video-per-core.")
    ap.add_argument("--cols", type=int, default=DEFAULT_COLS)
    ap.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    ap.add_argument("--skip", type=int, default=0, help="0: use N center frames; 1: use 2N from center & take every other; etc.")
    ap.add_argument("--invert_views", type=int, default=0, help="0: keep 1-D view order; 1: reverse the 1-D view order.")
    ap.add_argument("--in_dir", type=str, default=str(IN_DIR))
    ap.add_argument("--out_dir", type=str, default=str(OUT_DIR))
    ap.add_argument("--repair_before_merge", type=int, default=1, help="Re-encode left/right to concat-friendly format before merging (default 1).")
    ap.add_argument("--pre_focus", type=float, default=DEFAULT_PRE_FOCUS, help="Pre-apply horizontal focus as a percentage in [-1,1]; -1 = max left, +1 = max right; 0 disables.")
    ap.add_argument("--jobs", type=int, default=cpu_count(), help="Number of parallel workers. Defaults to #cores.")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cols, rows = args.cols, args.rows
    step = max(1, args.skip + 1)
    invert_views = int(args.invert_views)
    do_repair = int(args.repair_before_merge)
    pre_focus = max(-1.0, min(1.0, float(args.pre_focus)))
    jobs = max(1, int(args.jobs))

    pairs = find_lr_pairs(in_dir)
    if pairs:
        print(f"Found {len(pairs)} left/right pair(s) to merge.")
        merge_tasks = []
        for left, right, base in pairs:
            merged = in_dir / f"{base}.mp4"
            merge_tasks.append((str(left), str(right), str(merged), do_repair))
        with Pool(processes=jobs) as pool:
            for ok, msg in pool.imap_unordered(merge_pair_task, merge_tasks, chunksize=1):
                print(("OK  " if ok else "FAIL") + " " + msg)
    else:
        print("No left/right pairs to merge.")

    all_vids = [p for p in in_dir.iterdir() if is_video(p)]
    finals = [p for p in all_vids if is_final_output_video(p)]
    if not finals:
        print(f"No final videos found in {in_dir}.")
        return

    print("Selected final videos:")
    for p in finals:
        print("  -", p.name)

    quilt_tasks = []
    for src in finals:
        quilt_tasks.append((str(src), cols, rows, step, invert_views, pre_focus, FINAL_OUT_W, FINAL_OUT_H, str(out_dir)))

    with Pool(processes=jobs) as pool:
        for ok, msg in pool.imap_unordered(quilt_task, quilt_tasks, chunksize=1):
            print(("OK  " if ok else "FAIL") + " " + msg)

if __name__ == "__main__":
    main()
