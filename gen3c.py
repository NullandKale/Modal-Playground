# gen3c.py
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
        # keep container's torch stack; install the rest (strip TE/FA/Megatron)
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
    """
    One high-quality encode pass to scale & set DAR.
    H.264 CRF 12 + preset slow + yuv420p for compatibility.
    """
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
    if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
        _log(f"Cloning repo into {REPO_DIR} (first run)...")
        _run(["git", "clone", "--recurse-submodules", "https://github.com/nullandkale/GEN3C", REPO_DIR])
        return
    _run(["git", "-C", REPO_DIR, "pull", "--rebase", "--autostash"])
    _run(["git", "-C", REPO_DIR, "submodule", "update", "--init", "--recursive"])
    try:
        rev = _run(["git", "-C", REPO_DIR, "rev-parse", "--short", "HEAD"]).strip().splitlines()[-1]
        _log(f"Repo at {REPO_DIR} now on commit {rev}")
    except Exception:
        pass

# ---------- Render one trajectory ----------
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
    trajectory: str,                 # "left" or "right"
    num_video_frames: int,
    guidance: float,
    movement_distance: float,
    use_offload: bool
) -> Dict[str, str]:
    _log(f"Worker started for {base_name} [trajectory={trajectory}]")

    _git_pull_repo()
    _ensure_checkpoints("HUGGINGFACE_TOKEN")

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    stem = f"{base_name}_{trajectory}"
    raw_name = f"{stem}_rail"
    conditioning_stem = f"{stem}_input"
    depth_dir = os.path.join(OUTPUTS_DIR, "depth", stem)
    mask_dir = os.path.join(OUTPUTS_DIR, "mask", stem)
    buffers_zip = os.path.join(OUTPUTS_DIR, f"{stem}_buffers.zip")
    final_mp4 = os.path.join(OUTPUTS_DIR, f"{stem}.mp4")
    conditioning_mp4 = os.path.join(OUTPUTS_DIR, f"{conditioning_stem}.mp4")

    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

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
            "--camera_rotation", "no_rotation",
            "--movement_distance", str(movement_distance),
            "--num_video_frames", str(num_video_frames),
            "--save_conditioning_video",
            "--conditioning_video_name", conditioning_stem,
            "--save_depth_dir", depth_dir,
            "--save_mask_dir", mask_dir,
            "--disable_guardrail",
            "--disable_prompt_upsampler",
            "--disable_prompt_encoder",
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
            raise FileNotFoundError(f"Expected rail output missing: {raw_mp4}")

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
            # still rewrap to high-quality settings in one pass (in case source params differ)
            _ffmpeg_scale_only(raw_mp4, final_mp4, rw, rh, iw, ih)
            try:
                os.remove(raw_mp4)
            except OSError:
                pass

        def _add_tree(zf: zipfile.ZipFile, root_dir: str, prefix_in_zip: str):
            if os.path.isdir(root_dir):
                for p in sorted(glob.glob(os.path.join(root_dir, "**", "*"), recursive=True)):
                    if os.path.isfile(p):
                        arc = os.path.join(prefix_in_zip, os.path.relpath(p, root_dir))
                        zf.write(p, arcname=arc)

        with zipfile.ZipFile(buffers_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            _add_tree(zf, depth_dir, f"depth/{stem}")
            _add_tree(zf, mask_dir, f"mask/{stem}")

        outputs_vol.commit()

        _log(f"Done: {final_mp4}")
        return {
            "final": os.path.basename(final_mp4),
            "conditioning": os.path.basename(conditioning_mp4) if os.path.exists(conditioning_mp4) else "",
            "buffers": os.path.basename(buffers_zip) if os.path.exists(buffers_zip) else "",
        }
    finally:
        try:
            os.remove(tmp_img)
        except OSError:
            pass

# ---------- Merge left+right into one centered clip ----------
@stub.function(
    image=image,
    gpu=None,
    timeout=30 * 60,
    volumes={"/vol/outputs": outputs_vol},
)
def merge_left_right(base_name: str) -> str:
    """
    Single-pass high-quality merge:
      [left] --reverse--> [v0]
      [v0] + [right] --concat--> merged
    """
    left = os.path.join(OUTPUTS_DIR, f"{base_name}_left.mp4")
    right = os.path.join(OUTPUTS_DIR, f"{base_name}_right.mp4")
    merged = os.path.join(OUTPUTS_DIR, f"{base_name}.mp4")

    def exists(p: str) -> bool:
        ok = os.path.exists(p) and os.path.getsize(p) > 0
        _log(f"check {p}: {'OK' if ok else 'MISSING'}")
        return ok

    if not (exists(left) and exists(right)):
        _log("Left/right clips not present; skipping merge.")
        return ""

    try:
        # One encode, no intermediate files: reverse + concat, then output at high quality.
        _run([
            "ffmpeg", "-y",
            "-i", left,
            "-i", right,
            "-filter_complex", "[0:v]reverse[v0];[v0][1:v]concat=n=2:v=1:a=0[v]",
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

# ---------- Orchestrate one image end-to-end (left+right, merge) ----------
@stub.function(
    image=image,
    gpu=None,
    timeout=60 * 60,
    volumes={"/vol/outputs": outputs_vol, "/vol/weights": weights_vol},
)
def process_one_image(
    input_image_bytes: bytes,
    base_name: str,
    num_video_frames: int,
    guidance: float,
    movement_distance: float,
    use_offload: bool,
) -> Dict[str, str]:
    left_call = gen3c_render_single_move.spawn(
        input_image_bytes=input_image_bytes,
        base_name=base_name,
        trajectory="left",
        num_video_frames=num_video_frames,
        guidance=guidance,
        movement_distance=movement_distance,
        use_offload=use_offload,
    )
    right_call = gen3c_render_single_move.spawn(
        input_image_bytes=input_image_bytes,
        base_name=base_name,
        trajectory="right",
        num_video_frames=num_video_frames,
        guidance=guidance,
        movement_distance=movement_distance,
        use_offload=use_offload,
    )

    left_manifest = left_call.get()
    right_manifest = right_call.get()

    merged_basename = merge_left_right.remote(base_name)

    return {
        "base": base_name,
        "left_final": left_manifest.get("final", ""),
        "right_final": right_manifest.get("final", ""),
        "merged_final": merged_basename or "",
        "left_conditioning": left_manifest.get("conditioning", ""),
        "right_conditioning": right_manifest.get("conditioning", ""),
        "left_buffers": left_manifest.get("buffers", ""),
        "right_buffers": right_manifest.get("buffers", ""),
    }

@stub.local_entrypoint()
def main(
    input_image_path: str = "",
    output_dir: str = "output",
    frames: int = 121,
    guidance: float = 1.0,
    movement_distance: float = 0.5,
    offload: int = 0,
    max_concurrent_images: int = 20,
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
            num_video_frames=frames,
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
            left_final = manifest.get("left_final") or ""
            right_final = manifest.get("right_final") or ""
            if merged:
                _fetch(merged, out_dir / merged, "merged video")
                if (out_dir / merged).name != f"{base0}.mp4":
                    try:
                        shutil.copy2(out_dir / merged, out_dir / f"{base0}.mp4")
                        print(f"[{_now()}] Saved convenience copy: {out_dir / f'{base0}.mp4'}")
                    except Exception:
                        pass
            if left_final:
                _fetch(left_final, out_dir / left_final, "left final")
            if right_final:
                _fetch(right_final, out_dir / right_final, "right final")
            for key_name, label in [
                ("left_conditioning", "left conditioning"),
                ("right_conditioning", "right conditioning"),
                ("left_buffers", "left buffers"),
                ("right_buffers", "right buffers"),
            ]:
                k = manifest.get(key_name) or ""
                if k:
                    _fetch(k, out_dir / k, label)

    while active:
        call, base, _ = active.pop(0)
        try:
            manifest = call.get()
        except Exception as e:
            print(f"[{_now()}] ERROR: image '{base}' failed: {e}", file=sys.stderr)
            continue
        merged = manifest.get("merged_final") or ""
        left_final = manifest.get("left_final") or ""
        right_final = manifest.get("right_final") or ""
        if merged:
            _fetch(merged, out_dir / merged, "merged video")
            if (out_dir / merged).name != f"{base}.mp4":
                try:
                    shutil.copy2(out_dir / merged, out_dir / f"{base}.mp4")
                    print(f"[{_now()}] Saved convenience copy: {out_dir / f'{base}.mp4'}")
                except Exception:
                    pass
        if left_final:
            _fetch(left_final, out_dir / left_final, "left final")
        if right_final:
            _fetch(right_final, out_dir / right_final, "right final")
        for key_name, label in [
            ("left_conditioning", "left conditioning"),
            ("right_conditioning", "right conditioning"),
            ("left_buffers", "left buffers"),
            ("right_buffers", "right buffers"),
        ]:
            k = manifest.get(key_name) or ""
            if k:
                _fetch(k, out_dir / k, label)

    print(f"[{_now()}] Done.")
