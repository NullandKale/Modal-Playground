# gen3c.py
import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import modal

APP_NAME = "gen3c"
VOLUME_WEIGHTS_NAME = "gen3c-weights"
VOLUME_OUTPUTS_NAME = "gen3c-outputs"
HF_SECRET_NAME = "my-huggingface-secret"

REPO_DIR = "/opt/GEN3C"
WEIGHTS_DIR = "/vol/weights/checkpoints"
OUTPUTS_DIR = "/vol/outputs"

DEFAULT_GPU = "A100-80GB"
FUNC_TIMEOUT = 4 * 60 * 60

stub = modal.App(APP_NAME)
weights_vol = modal.Volume.from_name(VOLUME_WEIGHTS_NAME, create_if_missing=True)
outputs_vol = modal.Volume.from_name(VOLUME_OUTPUTS_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:24.10-py3")
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
        "TORCH_BACKEND_CUDNN_ALLOW_TF32": "1",
        "AIOHTTP_NO_EXTENSIONS": "1",
        "MULTIDICT_NO_EXTENSIONS": "1",
        "YARL_NO_EXTENSIONS": "1",
        "PIP_PREFER_BINARY": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    .apt_install("git", "ffmpeg", "build-essential", "clang", "ninja-build", "ca-certificates")
    .run_commands([
        f"test -d {REPO_DIR} || git clone --recurse-submodules https://github.com/nv-tlabs/GEN3C {REPO_DIR}",
        f"cd {REPO_DIR} && git submodule update --init --recursive",
        f"ls -la {REPO_DIR}",
        # keep container's torch; install the rest
        f"REQ={REPO_DIR}/requirements.txt; REQ2=/tmp/req-no-torch.txt; "
        "if [ -f $REQ ]; then grep -vE '^(torch|torchvision|torchaudio)($|[=<>])' $REQ > $REQ2; fi",
        "python3 -m pip install -r /tmp/req-no-torch.txt || true",
    ])
    .pip_install(
        "huggingface_hub>=0.34.0,<1.0",
        "requests>=2.31.0",
        "git+https://github.com/microsoft/MoGe.git",
        "trimesh>=4.0.0",
        "plyfile>=1.0.0",
        "numpy<2.0",
        "opencv-python==4.8.0.74",
        "imageio>=2.34.0",
        "imageio-ffmpeg>=0.4.9",
        "omegaconf>=2.3.0",
        "hydra-core>=1.3.2",
        "protobuf<5",
    )
)

def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def _log(msg: str) -> None:
    print(f"[{_now()}] {msg}", flush=True)

def _run(cmd: List[str], cwd: Optional[str] = None, extra_env: Optional[Dict[str, str]] = None) -> str:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    _log("START: " + " ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines: List[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line.rstrip("\n"), flush=True)
        lines.append(line)
    proc.stdout.close()
    rc = proc.wait()
    if rc != 0:
        tail = "".join(lines).splitlines()[-120:]
        raise RuntimeError("Subprocess failed:\n" + "\n".join(tail))
    _log("DONE : " + " ".join(cmd))
    return "".join(lines)

def _ensure_checkpoints(hf_token_env_var: str) -> None:
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    sentinel = os.path.join(WEIGHTS_DIR, "Gen3C-Cosmos-7B", "model.pt")
    if os.path.exists(sentinel):
        _log(f"Checkpoints already present at {WEIGHTS_DIR}; skipping download.")
        return
    env = {
        "PYTHONPATH": REPO_DIR,
        "HUGGING_FACE_HUB_TOKEN": os.environ.get(hf_token_env_var, ""),
        "HF_HOME": "/vol/weights/hf-cache",
        "HUGGINGFACE_HUB_CACHE": "/vol/weights/hf-cache",
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
    }
    _log("Downloading GEN3C checkpoints into persistent Volume...")
    _run(["python", "-m", "scripts.download_gen3c_checkpoints", "--checkpoint_dir", WEIGHTS_DIR], cwd=REPO_DIR, extra_env=env)
    if not os.path.exists(sentinel):
        raise FileNotFoundError(f"Expected checkpoint not found after download: {sentinel}")
    _log(f"Checkpoints ready at {WEIGHTS_DIR}")

def _write_temp_png(data: bytes) -> str:
    import tempfile
    import numpy as _np
    import cv2 as _cv2
    fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    img_array = _np.frombuffer(data, dtype=_np.uint8)
    decoded = _cv2.imdecode(img_array, _cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.size == 0:
        raise ValueError("Failed to decode input image.")
    if not _cv2.imwrite(tmp_path, decoded):
        raise ValueError("Failed to encode temp PNG.")
    _log(f"Wrote temp input image (forced PNG): {tmp_path}")
    return tmp_path

def _resolve_output_path(video_name: str) -> str:
    import shutil
    import glob
    direct = os.path.join(OUTPUTS_DIR, video_name + ".mp4")
    if os.path.exists(direct):
        return direct
    candidates = [
        os.path.join(REPO_DIR, "outputs", video_name + ".mp4"),
        os.path.join(REPO_DIR, video_name + ".mp4"),
    ]
    for c in candidates:
        if os.path.exists(c):
            dst = os.path.join(OUTPUTS_DIR, video_name + ".mp4")
            os.makedirs(OUTPUTS_DIR, exist_ok=True)
            shutil.copyfile(c, dst)
            return dst
    hydra_patterns = [
        os.path.join(REPO_DIR, "outputs", "**", video_name + ".mp4"),
        os.path.join(REPO_DIR, "**", "outputs", "**", video_name + ".mp4"),
        os.path.join(REPO_DIR, "cosmos_predict1", "**", "outputs", "**", video_name + ".mp4"),
    ]
    for pattern in hydra_patterns:
        matches = sorted(glob.glob(pattern, recursive=True))
        for m in matches:
            if os.path.isfile(m):
                dst = os.path.join(OUTPUTS_DIR, video_name + ".mp4")
                os.makedirs(OUTPUTS_DIR, exist_ok=True)
                shutil.copyfile(m, dst)
                return dst
    fallback_patterns = [
        os.path.join("/opt", "**", "outputs", "**", video_name + ".mp4"),
        os.path.join("/tmp", "**", video_name + ".mp4"),
        os.path.join("/root", "**", "outputs", "**", video_name + ".mp4"),
    ]
    for pattern in fallback_patterns:
        matches = sorted(glob.glob(pattern, recursive=True))
        for m in matches:
            if os.path.isfile(m):
                dst = os.path.join(OUTPUTS_DIR, video_name + ".mp4")
                os.makedirs(OUTPUTS_DIR, exist_ok=True)
                shutil.copyfile(m, dst)
                return dst
    raise FileNotFoundError(f"Expected output not found for {video_name}")

def _ffprobe_resolution(p: str) -> Tuple[int, int]:
    out = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", p])
    parts = out.strip().splitlines()[-1].split("x")
    return int(parts[0]), int(parts[1])

def _even(n: int) -> int:
    return n if n % 2 == 0 else n - 1

def _ffmpeg_scale_only(in_mp4: str, out_mp4: str, target_w: int, target_h: int, dar_num: int, dar_den: int) -> None:
    tmp = out_mp4 + ".tmp.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-i", in_mp4,
        "-vf", f"scale={target_w}:{target_h}:flags=lanczos,setdar={dar_num}/{dar_den}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        tmp,
    ]
    _run(cmd)
    os.replace(tmp, out_mp4)
    _log(f"Scaled to input aspect ratio: {out_mp4}")

# -------- Runtime patching: make 'left' a pure linear rail, start at +d -> -d, no rotation --------
def _patch_left_linear_rail(start_on_right: bool = True) -> None:
    """
    Edits GEN3C's camera_utils.py at runtime so that requesting trajectory='left'
    yields a *linear translation along X* from +d to -d (if start_on_right)
    or from -d to +d (if not). We patch dispatch robustly (dicts or wrappers).
    """
    cu_path = os.path.join(REPO_DIR, "cosmos_predict1", "diffusion", "inference", "camera_utils.py")
    if not os.path.exists(cu_path):
        _log("camera_utils.py not found; cannot patch.")
        return
    try:
        txt = Path(cu_path).read_text()
        if "# --- patched: left -> linear rail (runtime) ---" in txt:
            _log("Left->rail patch already present; skipping.")
            return

        direction = "movement_distance, -movement_distance" if start_on_right else "-movement_distance, movement_distance"
        patch = f"""

# --- patched: left -> linear rail (runtime) ---
def __gen3c_modal__linear_rail_poses(num_frames, movement_distance, device, start_on_right={str(start_on_right)}):
    import torch
    # translate along X: +d .. -d if start_on_right else -d .. +d
    if start_on_right:
        t = torch.linspace(movement_distance, -movement_distance, steps=num_frames, device=device)
    else:
        t = torch.linspace(-movement_distance, movement_distance, steps=num_frames, device=device)
    poses = torch.eye(4, device=device).repeat(num_frames, 1, 1)
    poses[:, 0, 3] = t
    return poses

def __gen3c_modal__patch_dispatch():
    g = globals()
    # If a dict-based dispatch exists, replace just 'left'
    for name in ("TRAJECTORY_DISPATCH", "TRAJECTORY_TO_FUNC", "TRAJECTORY_MAP", "TRAJECTORY_FUNCS"):
        d = g.get(name, None)
        if isinstance(d, dict) and "left" in d:
            def _f(*args, **kwargs):
                nf = kwargs.get("num_frames") or (len(args) > 1 and args[1]) or 121
                md = kwargs.get("movement_distance") or (len(args) > 2 and float(args[2])) or 0.5
                dev = kwargs.get("device")
                if dev is None:
                    try:
                        import torch
                        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    except Exception:
                        dev = None
                return __gen3c_modal__linear_rail_poses(int(nf), float(md), dev)
            d["left"] = _f
            return True

    # Otherwise, wrap common builders
    def _wrap(func):
        def _wrapped(*args, **kwargs):
            tr = kwargs.get("trajectory")
            if tr is None:
                for a in args:
                    if isinstance(a, str):
                        tr = a; break
            if tr != "left":
                return func(*args, **kwargs)
            nf = kwargs.get("num_frames")
            md = kwargs.get("movement_distance")
            dev = kwargs.get("device")
            # heuristics from positional args
            if nf is None:
                for a in args:
                    if isinstance(a, int):
                        nf = a; break
            if md is None:
                for a in args:
                    if isinstance(a, (int,float)) and not isinstance(a, bool):
                        md = float(a); break
            if dev is None:
                try:
                    import torch
                    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                except Exception:
                    dev = None
            if nf is None: nf = 121
            if md is None: md = 0.5
            return __gen3c_modal__linear_rail_poses(int(nf), float(md), dev)
        try:
            _wrapped.__name__ = func.__name__
        except Exception:
            pass
        return _wrapped

    g = globals()
    for fname in ("get_camera_poses", "build_camera_trajectory", "get_camera_trajectory", "generate_camera_trajectory"):
        if fname in g and callable(g[fname]):
            g[fname] = _wrap(g[fname])
            return True
    return False

__gen3c_modal__patch_dispatch()
# --- end patch ---

"""
        Path(cu_path).write_text(txt + patch)
        _log("Patched camera_utils.py: left movement now linear rail (starts on right).")
    except Exception as e:
        _log(f"Failed to patch camera_utils.py: {e}")

# --------------------------------- Single-pass rail runner ---------------------------------
@stub.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=FUNC_TIMEOUT,
    volumes={"/vol/weights": weights_vol, "/vol/outputs": outputs_vol},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
)
def gen3c_render_linear_rail(
    input_image_bytes: bytes,
    base_name: str,
    num_video_frames: int,
    guidance: float,
    movement_distance: float,
    use_offload: bool
) -> str:
    """
    One pass: patch 'left' trajectory at runtime into a linear rail (pure X translation),
    starting at +d → center → -d, and run with --camera_rotation no_rotation.
    """
    _log(f"Worker started (single pass) for {base_name} [left patched -> linear rail]")
    _ensure_checkpoints("HUGGINGFACE_TOKEN")

    # runtime patch (no image rebuild)
    _patch_left_linear_rail(start_on_right=True)

    # temp input
    tmp_img = _write_temp_png(input_image_bytes)
    try:
        env = {"PYTHONPATH": REPO_DIR}
        os.makedirs(OUTPUTS_DIR, exist_ok=True)

        raw_name = f"{base_name}_rail"
        cmd = [
            "python", f"{REPO_DIR}/cosmos_predict1/diffusion/inference/gen3c_single_image.py",
            "--checkpoint_dir", WEIGHTS_DIR,
            "--input_image_path", tmp_img,
            "--video_save_name", raw_name,
            "--guidance", str(guidance),
            "--foreground_masking",
            "--trajectory", "left",             # <- our patched 'left' is linear rail
            "--camera_rotation", "no_rotation", # <- keep camera fixed (no tracking)
            "--movement_distance", str(movement_distance),
            "--num_video_frames", str(num_video_frames),
        ]
        if use_offload:
            cmd += [
                "--offload_diffusion_transformer",
                "--offload_tokenizer",
                "--offload_text_encoder_model",
                "--offload_prompt_upsampler",
                "--offload_guardrail_models",
                "--disable_guardrail",
                "--disable_prompt_encoder",
            ]
        _log("BEGIN render (patched left -> rail, start +d)")
        _run(cmd, cwd=REPO_DIR, extra_env=env)

        raw_mp4 = _resolve_output_path(raw_name)
        _log(f"Render OK {base_name} [rail via left] -> {raw_mp4}")

        # match output width to input DAR (height preserved)
        import numpy as _np, cv2 as _cv2
        arr = _np.frombuffer(input_image_bytes, dtype=_np.uint8)
        img = _cv2.imdecode(arr, _cv2.IMREAD_UNCHANGED)
        if img is None or img.size == 0:
            raise ValueError("Failed to decode input image for aspect computation.")
        iw, ih = int(img.shape[1]), int(img.shape[0])
        rw, rh = _ffprobe_resolution(raw_mp4)
        target_h = rh if rh % 2 == 0 else rh - 1
        target_w = _even(round(target_h * (iw / ih)))

        final_mp4 = os.path.join(OUTPUTS_DIR, f"{base_name}.mp4")
        if target_w != rw or target_h != rh:
            _ffmpeg_scale_only(raw_mp4, final_mp4, target_w, target_h, iw, ih)
            try:
                os.remove(raw_mp4)
            except OSError:
                pass
        else:
            os.replace(raw_mp4, final_mp4)

        _log(f"Single-pass done: {final_mp4}")
        # Return the path relative to the *volume root*. Since the volume is mounted at /vol/outputs,
        # the file lives at the root of the volume under the basename.
        return os.path.basename(final_mp4)
    finally:
        try:
            os.remove(tmp_img)
        except OSError:
            pass

# ------------------------------ Batch entrypoint (local) -----------------------------------
def _gather_images(path_str: str) -> List[Path]:
    p = Path(path_str)
    if p.is_file():
        return [p]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    if p.is_dir():
        return sorted([q for q in p.iterdir() if q.is_file() and q.suffix.lower() in exts])
    return []

def _read_from_outputs_volume(key: str, *, retries: int = 6, delay: float = 0.8) -> bytes:
    """
    Modal Volume read can be eventually-consistent across processes. We:
      1) reload()
      2) attempt 'key' (e.g., 'foo.mp4')
      3) attempt 'outputs/key' in case the provider exposes the mount dir level
    """
    last_err = None
    for attempt in range(retries):
        try:
            outputs_vol.reload()
        except Exception:
            pass
        # Try plain key
        try:
            chunks = []
            for chunk in outputs_vol.read_file(key):
                chunks.append(chunk)
            if chunks:
                return b"".join(chunks)
        except FileNotFoundError as e:
            last_err = e
        # Try 'outputs/key'
        try:
            chunks = []
            for chunk in outputs_vol.read_file(f"outputs/{key}"):
                chunks.append(chunk)
            if chunks:
                return b"".join(chunks)
        except FileNotFoundError as e:
            last_err = e
        time.sleep(delay)
    raise FileNotFoundError(f"Could not read '{key}' from outputs volume after {retries} attempts. Last error: {last_err}")

@stub.local_entrypoint()
def main(
    input_image_path: str = "",
    output_dir: str = "output",
    frames: int = 121,
    guidance: float = 1.0,
    movement_distance: float = 0.5,
    offload: int = 0,
) -> None:
    """
    For each image under input path: patches 'left' at runtime into a right→left linear rail,
    renders one pass with no rotation, and saves <name>.mp4 locally.
    """
    if not input_image_path:
        input_image_path = "input"
    paths = _gather_images(input_image_path)
    if not paths:
        print(f"No input images found at: {input_image_path}", file=sys.stderr)
        sys.exit(2)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{_now()}] Found {len(paths)} image(s) under {input_image_path}")

    for p in paths:
        base_name = p.stem
        print(f"[{_now()}] Submitting: {p}")
        try:
            key = gen3c_render_linear_rail.remote(
                input_image_bytes=p.read_bytes(),
                base_name=base_name,
                num_video_frames=frames,
                guidance=guidance,
                movement_distance=movement_distance,
                use_offload=bool(offload),
            )
        except Exception as e:
            print(f"Remote execution failed: {e}", file=sys.stderr)
            sys.exit(1)

        local_out = out_dir / f"{base_name}.mp4"
        try:
            data = _read_from_outputs_volume(key)
        except Exception as e:
            print(f"[{_now()}] ERROR reading from outputs volume for {base_name}: {e}", file=sys.stderr)
            sys.exit(1)
        with open(local_out, "wb") as f:
            f.write(data)
        print(f"[{_now()}] Saved: {local_out}")

    print(f"[{_now()}] Done.")
