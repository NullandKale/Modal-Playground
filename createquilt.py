# createquilt.py
import os
import json
import math
import subprocess
from pathlib import Path

# ---------- SETTINGS ----------
# Default single preset: 16" Light Field Display (Landscape) -> 7 x 7 grid
DEFAULT_COLS = 7
DEFAULT_ROWS = 7

# Optional: after tiling, scale the quilt to this final resolution.
# Using the recommended 16" Landscape quilt size (square). Set to None to skip scaling.
FINAL_OUT_W = 5999
FINAL_OUT_H = 5999
# ------------------------------

IN_DIR = Path("output")
OUT_DIR = Path("output_quilts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_EXTS = {".mp4", ".mov", ".webm", ".m4v"}

def run(cmd):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out_lines = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line.rstrip())
        out_lines.append(line)
    rc = proc.wait()
    if rc != 0:
        tail = "".join(out_lines).splitlines()[-80:]
        raise RuntimeError("Command failed:\n" + "\n".join(tail))

def ffprobe_json(path: Path):
    cmd = [
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_streams", "-show_format",
        str(path)
    ]
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)

def get_video_wh_and_frames(path: Path):
    data = ffprobe_json(path)
    vstreams = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
    if not vstreams:
        raise ValueError(f"No video stream in {path}")
    vs = vstreams[0]
    w = int(vs.get("width"))
    h = int(vs.get("height"))
    # nb_frames can be missing; if so, estimate from duration * avg_frame_rate
    nb_frames = None
    if "nb_frames" in vs and str(vs["nb_frames"]).isdigit():
        nb_frames = int(vs["nb_frames"])
    else:
        dur = None
        if "duration" in vs:
            try: dur = float(vs["duration"])
            except: pass
        if dur is None:
            try: dur = float(data.get("format", {}).get("duration", "0"))
            except: dur = 0.0
        afr = vs.get("avg_frame_rate", "0/1")
        try:
            num, den = afr.split("/")
            afr_val = float(num) / float(den) if float(den) != 0 else 0.0
        except:
            afr_val = 0.0
        if afr_val > 0 and dur > 0:
            nb_frames = int(math.floor(dur * afr_val))
    return w, h, nb_frames

def format_aspect(a):
    # Name requires decimal "a{aspect}". Use video DAR = width/height.
    # Keep up to 4 decimal places, strip trailing zeros.
    s = f"{a:.4f}"
    s = s.rstrip("0").rstrip(".")
    if s == "":
        s = "0"
    return s

def build_filter(cols, rows, take_frames, out_w=None, out_h=None):
    """
    Quilt spec requires bottom-left to top-right ordering; ffmpeg tile fills
    top-left -> bottom-right. Double vflip corrects that (docs).
    We trim to the first N = cols*rows frames, REVERSE that set, then tile them.
    """
    parts = []
    # take first `take_frames` frames
    parts.append(f"trim=end_frame={take_frames}")
    # reverse just those frames' order
    parts.append("reverse")
    # double vertical flip to convert ffmpeg's top-left ordering into quilt bottom-left ordering
    parts.append("vflip")
    parts.append(f"tile={cols}x{rows}")
    parts.append("vflip")
    if out_w and out_h:
        parts.append(f"scale={out_w}:{out_h}:flags=lanczos")
    # join with commas (single filter chain)
    return ",".join(parts)

def main():
    items = [p for p in IN_DIR.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    if not items:
        print(f"No videos in {IN_DIR}")
        return

    cols, rows = DEFAULT_COLS, DEFAULT_ROWS
    view_total = cols * rows

    for src in items:
        try:
            print(f"\n=== Processing: {src.name} ===")
            w, h, nb = get_video_wh_and_frames(src)
            if nb is not None and nb < view_total:
                print(f"Warning: {src.name} has only {nb} frames, but quilt needs {view_total}. Proceeding with first {view_total} (ffmpeg will drop if insufficient).")

            aspect = w / float(h) if h else 1.0
            a_str = format_aspect(aspect)

            base = src.stem
            out_name = f"{base}_qs{cols}x{rows}a{a_str}.png"
            out_path = OUT_DIR / out_name

            vf = build_filter(cols, rows, view_total, FINAL_OUT_W, FINAL_OUT_H)

            cmd = [
                "ffmpeg", "-y",
                "-i", str(src),
                "-filter_complex", vf,
                "-frames:v", "1",
                str(out_path)
            ]
            run(cmd)
            print(f"Saved quilt: {out_path}")
        except Exception as e:
            print(f"ERROR on {src.name}: {e}")

if __name__ == "__main__":
    main()
