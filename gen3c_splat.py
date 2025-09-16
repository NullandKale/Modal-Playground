# gen3c_splat.py
import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import shutil
import glob
import zipfile

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

# ---------------- Fixed single-image script expectations ----------------
NUM_FRAMES = 121
# -----------------------------------------------------------------------

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
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": f"{REPO_DIR}",
    })
    .apt_install("git", "ffmpeg", "build-essential", "clang", "ninja-build", "ca-certificates")
    .run_commands([
        f"test -d {REPO_DIR} || git clone --recurse-submodules https://github.com/nullandkale/GEN3C {REPO_DIR}",
        f"cd {REPO_DIR} && git submodule update --init --recursive",
        f"ls -la {REPO_DIR}",
        f"REQ={REPO_DIR}/requirements.txt; REQ2=/tmp/req-pruned.txt; "
        "if [ -f $REQ ]; then grep -viE '^(torch|torchvision|torchaudio|transformer[-_]?engine|flash[-_]?attn|flash[-_]?attention)($|[=<>])' $REQ > $REQ2; else : > $REQ2; fi",
        "python3 -m pip install -r /tmp/req-pruned.txt || true",
    ])
    .pip_install(
        "diffusers>=0.33.0,<0.36",
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
        "peft>=0.17.0",
    )
)

def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def _log(msg: str) -> None:
    print(f"[{_now()}] {msg}", flush=True)

def _maybe_wrap_unbuffered(cmd: List[str]) -> List[str]:
    new_cmd = list(cmd)
    if new_cmd:
        exe = os.path.basename(new_cmd[0])
        if exe.startswith("python"):
            if "-u" not in new_cmd[1:3]:
                new_cmd = [new_cmd[0], "-u"] + new_cmd[1:]
        else:
            stdbuf = shutil.which("stdbuf")
            if stdbuf and exe != "stdbuf":
                new_cmd = [stdbuf, "-oL", "-eL"] + new_cmd
    return new_cmd

def _run(cmd: List[str], cwd: Optional[str] = None, extra_env: Optional[Dict[str, str]] = None) -> str:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    run_cmd = _maybe_wrap_unbuffered(cmd)
    _log("START: " + " ".join(run_cmd))
    proc = subprocess.Popen(run_cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
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
    _log("DONE : " + " ".join(run_cmd))
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

def _ffprobe_resolution(p: str) -> Tuple[int, int]:
    out = _run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0", p
    ])
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
        "-c:v", "libx264", "-preset", "slow", "-crf", "12",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        tmp,
    ]
    _run(cmd)
    os.replace(tmp, out_mp4)

def _gather_images(path_str: str) -> List[Path]:
    p = Path(path_str)
    if p.is_file():
        return [p]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    if p.is_dir():
        return sorted([q for q in p.iterdir() if q.is_file() and q.suffix.lower() in exts])
    return []

def _read_from_outputs_volume(key: str, *, retries: int = 6, delay: float = 0.8) -> bytes:
    last_err = None
    for _ in range(retries):
        try:
            outputs_vol.reload()
        except Exception:
            pass
        try:
            chunks = []
            for chunk in outputs_vol.read_file(key):
                chunks.append(chunk)
            if chunks:
                return b"".join(chunks)
        except FileNotFoundError as e:
            last_err = e
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

def _git_pull_repo() -> None:
    url = "https://github.com/nv-tlabs/GEN3C"
    if os.path.isdir(REPO_DIR):
        _log(f"Removing {REPO_DIR} ...")
        shutil.rmtree(REPO_DIR, ignore_errors=True)
    _log(f"Cloning fresh repo into {REPO_DIR} from {url} ...")
    _run(["git", "clone", "--recurse-submodules", url, REPO_DIR])
    _run(["git", "-C", REPO_DIR, "submodule", "update", "--init", "--recursive"])
    try:
        rev = _run(["git", "-C", REPO_DIR, "rev-parse", "--short", "HEAD"]).strip().splitlines()[-1]
        _log(f"Repo at {REPO_DIR} now on commit {rev}")
    except Exception:
        pass

# ---------- Render one trajectory (121 frames) ----------
@stub.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=FUNC_TIMEOUT,
    volumes={"/vol/weights": weights_vol, "/vol/outputs": outputs_vol},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
)
def gen3c_render_single_move(
    input_image_bytes: bytes,
    base_name: str,
    trajectory: str,
    guidance: float,
    movement_distance: float,
    use_offload: bool,
    camera_rotation: str
) -> Dict[str, str]:
    _log(f"Worker started for {base_name} [trajectory={trajectory}, camera_rotation={camera_rotation}]")
    _git_pull_repo()
    _ensure_checkpoints("HUGGINGFACE_TOKEN")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    stem = f"{base_name}_{trajectory}"
    raw_name = f"{stem}_rail"
    final_mp4 = os.path.join(OUTPUTS_DIR, f"{stem}.mp4")
    tmp_img = _write_temp_png(input_image_bytes)
    try:
        env = {"PYTHONPATH": REPO_DIR}
        cmd = [
            "python", f"{REPO_DIR}/cosmos_predict1/diffusion/inference/gen3c_single_image.py",
            "--checkpoint_dir", WEIGHTS_DIR,
            "--input_image_path", tmp_img,
            "--video_save_name", raw_name,
            "--video_save_folder", OUTPUTS_DIR,
            "--guidance", str(guidance),
            "--foreground_masking",
            "--trajectory", trajectory,
            "--camera_rotation", camera_rotation,
            "--movement_distance", str(movement_distance),
            "--num_video_frames", str(NUM_FRAMES),
        ]
        if use_offload:
            cmd += [
                "--offload_diffusion_transformer",
                "--offload_tokenizer",
                "--offload_text_encoder_model",
                "--offload_prompt_upsampler",
                "--offload_guardrail_models",
            ]
        _log(f"BEGIN render ({trajectory})")
        _run(cmd, cwd=REPO_DIR, extra_env=env)
        raw_mp4 = os.path.join(OUTPUTS_DIR, raw_name + ".mp4")
        if not os.path.exists(raw_mp4):
            raise FileNotFoundError(f"Expected output missing: {raw_mp4}")
        import numpy as _np, cv2 as _cv2
        arr = _np.frombuffer(input_image_bytes, dtype=_np.uint8)
        img = _cv2.imdecode(arr, _cv2.IMREAD_UNCHANGED)
        if img is None or img.size == 0:
            raise ValueError("Failed to decode input image for aspect computation.")
        iw, ih = int(img.shape[1]), int(img.shape[0])
        rw, rh = _ffprobe_resolution(raw_mp4)
        target_h = rh if rh % 2 == 0 else rh - 1
        target_w = _even(round(target_h * (iw / ih)))
        if target_w != rw or target_h != rh:
            _ffmpeg_scale_only(raw_mp4, final_mp4, target_w, target_h, iw, ih)
            try:
                os.remove(raw_mp4)
            except OSError:
                pass
        else:
            _ffmpeg_scale_only(raw_mp4, final_mp4, rw, rh, iw, ih)
            try:
                os.remove(raw_mp4)
            except OSError:
                pass
        outputs_vol.commit()
        _log(f"Done: {final_mp4}")
        return {
            "final": os.path.basename(final_mp4),
        }
    finally:
        try:
            os.remove(tmp_img)
        except OSError:
            pass

# ---------- Merge three clips into one sequence ----------
@stub.function(
    image=image,
    gpu=None,
    timeout=30 * 60,
    volumes={"/vol/outputs": outputs_vol},
)
def merge_three_movements(base_name: str) -> str:
    """
    Concatenate: left + right + zoom-out
    """
    left = os.path.join(OUTPUTS_DIR, f"{base_name}_left.mp4")
    right = os.path.join(OUTPUTS_DIR, f"{base_name}_right.mp4")
    zoom = os.path.join(OUTPUTS_DIR, f"{base_name}_zoom_out.mp4")
    merged = os.path.join(OUTPUTS_DIR, f"{base_name}.mp4")
    def exists(p: str) -> bool:
        ok = os.path.exists(p) and os.path.getsize(p) > 0
        _log(f"check {p}: {'OK' if ok else 'MISSING'}")
        return ok
    if not (exists(left) and exists(right) and exists(zoom)):
        _log("One or more movement clips not present; skipping merge.")
        return ""
    try:
        _run([
            "ffmpeg", "-y",
            "-i", left,
            "-i", right,
            "-i", zoom,
            "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
            "-map", "[v]",
            "-c:v", "libx264", "-preset", "slow", "-crf", "12",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            merged,
        ])
        outputs_vol.commit()
        _log(f"Merged to {merged}")
        return os.path.basename(merged)
    except Exception as e:
        _log(f"[WARN] Merge failed: {e}")
        try:
            outputs_vol.commit()
        except Exception:
            pass
        return ""

# ---------- Orchestrate one image end-to-end (left, right, zoom-out, merge) ----------
@stub.function(
    image=image,
    gpu=None,
    timeout=60 * 60,
    volumes={"/vol/outputs": outputs_vol, "/vol/weights": weights_vol},
)
def process_one_image(
    input_image_bytes: bytes,
    base_name: str,
    guidance: float,
    movement_distance: float,
    use_offload: bool,
) -> Dict[str, str]:
    rot_left_call = gen3c_render_single_move.spawn(
        input_image_bytes=input_image_bytes,
        base_name=base_name,
        trajectory="left",
        guidance=guidance,
        movement_distance=movement_distance,
        use_offload=use_offload,
        camera_rotation="center_facing",
    )
    rot_right_call = gen3c_render_single_move.spawn(
        input_image_bytes=input_image_bytes,
        base_name=base_name,
        trajectory="right",
        guidance=guidance,
        movement_distance=movement_distance,
        use_offload=use_offload,
        camera_rotation="center_facing",
    )
    zoom_out_call = gen3c_render_single_move.spawn(
        input_image_bytes=input_image_bytes,
        base_name=base_name,
        trajectory="zoom_out",
        guidance=guidance,
        movement_distance=movement_distance,
        use_offload=use_offload,
        camera_rotation="center_facing",
    )
    rot_left_manifest = rot_left_call.get()
    rot_right_manifest = rot_right_call.get()
    zoom_out_manifest = zoom_out_call.get()
    merged_basename = merge_three_movements.remote(base_name)
    return {
        "base": base_name,
        "rotate_left_final": rot_left_manifest.get("final", ""),
        "rotate_right_final": rot_right_manifest.get("final", ""),
        "zoom_out_final": zoom_out_manifest.get("final", ""),
        "merged_final": merged_basename or "",
    }

@stub.local_entrypoint()
def main(
    input_image_path: str = "",
    output_dir: str = "output",
    guidance: float = 1.0,
    movement_distance: float = 0.25,
    offload: int = 0,
    max_concurrent_images: int = 1,
) -> None:
    if not input_image_path:
        input_image_path = "input"
    paths = _gather_images(input_image_path)
    if not paths:
        print(f"No input images found at: {input_image_path}", file=sys.stderr)
        sys.exit(2)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{_now()}] Found {len(paths)} image(s) under {input_image_path}")
    print(f"[{_now()}] NUM_FRAMES={NUM_FRAMES}")
    print(f"[{_now()}] Movements per image: left + right + zoom-out -> merged")
    def _fetch(key: str, local_path: Path, label: str) -> bool:
        if not key:
            return False
        try:
            data = _read_from_outputs_volume(key)
            with open(local_path, "wb") as f:
                f.write(data)
            print(f"[{_now()}] Saved: {local_path}")
            return True
        except Exception as e:
            print(f"[{_now()}] WARNING: could not read {label} ({e})")
            return False
    active: List[Tuple[modal.FunctionCall, str, bytes]] = []
    for p in paths:
        base = p.stem
        img_bytes = p.read_bytes()
        print(f"[{_now()}] Submitting: {p} (base={base})")
        call = process_one_image.spawn(
            input_image_bytes=img_bytes,
            base_name=base,
            guidance=guidance,
            movement_distance=movement_distance,
            use_offload=bool(offload),
        )
        active.append((call, base, img_bytes))
        if len(active) >= max_concurrent_images:
            call0, base0, _ = active.pop(0)
            try:
                manifest = call0.get()
            except Exception as e:
                print(f"[{_now()}] ERROR: image '{base0}' failed: {e}", file=sys.stderr)
                continue
            merged = manifest.get("merged_final") or ""
            rotl = manifest.get("rotate_left_final") or ""
            rotr = manifest.get("rotate_right_final") or ""
            zoomo = manifest.get("zoom_out_final") or ""
            if merged:
                _fetch(merged, out_dir / merged, "merged video")
                if (out_dir / merged).name != f"{base0}.mp4":
                    try:
                        shutil.copy2(out_dir / merged, out_dir / f"{base0}.mp4")
                        print(f"[{_now()}] Saved convenience copy: {out_dir / f'{base0}.mp4'}")
                    except Exception:
                        pass
            if rotl:
                _fetch(rotl, out_dir / rotl, "rotate-left final")
            if rotr:
                _fetch(rotr, out_dir / rotr, "rotate-right final")
            if zoomo:
                _fetch(zoomo, out_dir / zoomo, "zoom-out final")
    while active:
        call, base, _ = active.pop(0)
        try:
            manifest = call.get()
        except Exception as e:
            print(f"[{_now()}] ERROR: image '{base}' failed: {e}", file=sys.stderr)
            continue
        merged = manifest.get("merged_final") or ""
        rotl = manifest.get("rotate_left_final") or ""
        rotr = manifest.get("rotate_right_final") or ""
        zoomo = manifest.get("zoom_out_final") or ""
        if merged:
            _fetch(merged, out_dir / merged, "merged video")
            if (out_dir / merged).name != f"{base}.mp4":
                try:
                    shutil.copy2(out_dir / merged, out_dir / f"{base}.mp4")
                    print(f"[{_now()}] Saved convenience copy: {out_dir / f'{base}.mp4'}")
                except Exception:
                    pass
        if rotl:
            _fetch(rotl, out_dir / rotl, "rotate-left final")
        if rotr:
            _fetch(rotr, out_dir / rotr, "rotate-right final")
        if zoomo:
            _fetch(zoomo, out_dir / zoomo, "zoom-out final")
    print(f"[{_now()}] Done.")

