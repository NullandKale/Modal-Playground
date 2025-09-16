import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import shutil
import re
import tempfile

import modal

APP_NAME = "gen3c-quilt"
VOLUME_WEIGHTS_NAME = "gen3c-weights"
VOLUME_OUTPUTS_NAME = "gen3c-outputs"
HF_SECRET_NAME = "my-huggingface-secret"

REPO_DIR = "/opt/GEN3C"
WEIGHTS_DIR = "/vol/weights/checkpoints"
OUTPUTS_DIR = "/vol/outputs"

# Target resolution (multiples of 64 to be safe)
TARGET_W = 1280
TARGET_H = 704

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
        f"REQ={REPO_DIR}/requirements.txt; REQ2=/tmp/req-pruned.txt; if [ -f $REQ ]; then grep -viE '^(torch|torchvision|torchaudio|transformer[-_]?engine|flash[-_]?attn|flash[-_]?attention)($|[=<>])' $REQ > $REQ2; else : > $REQ2; fi",
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
    env = {"PYTHONPATH": REPO_DIR, "HUGGING_FACE_HUB_TOKEN": os.environ.get(hf_token_env_var, ""), "HF_HOME": "/vol/weights/hf-cache", "HUGGINGFACE_HUB_CACHE": "/vol/weights/hf-cache", "HF_HUB_ENABLE_HF_TRANSFER": "0"}
    _log("Downloading GEN3C checkpoints into persistent Volume...")
    _run(["python", "-m", "scripts.download_gen3c_checkpoints", "--checkpoint_dir", WEIGHTS_DIR], cwd=REPO_DIR, extra_env=env)
    if not os.path.exists(sentinel):
        raise FileNotFoundError(f"Expected checkpoint not found after download: {sentinel}")
    _log(f"Checkpoints ready at {WEIGHTS_DIR}")

def _gather_quilts(path_str: str) -> List[Path]:
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

def _parse_quilt_meta_from_name(filename: str) -> Tuple[int, int]:
    m = re.search(r"_qs(?P<c>\d+)x(?P<r>\d+)(?:a(?P<a>[0-9]*\.?[0-9]+))?", Path(filename).stem, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Filename does not encode quilt meta via _qsCxRaAR: {filename}")
    return int(m.group("c")), int(m.group("r"))

def _write_temp_png(data: bytes) -> str:
    fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    import numpy as _np
    import cv2 as _cv2
    img_array = _np.frombuffer(data, dtype=_np.uint8)
    decoded = _cv2.imdecode(img_array, _cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.size == 0:
        raise ValueError("Failed to decode input quilt image.")
    if not _cv2.imwrite(tmp_path, decoded):
        raise ValueError("Failed to encode temp PNG.")
    _log(f"Wrote temp quilt image (forced PNG): {tmp_path}")
    return tmp_path

def _write_temp_bytes(data: bytes, suffix: str) -> str:
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    _log(f"Wrote temp sidecar: {tmp_path}")
    return tmp_path

def _find_sidecars_for_quilt(path: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Returns (meta_json_path, center_depth_png16_path) if present, else (None, None).
    Sidecar naming matches rgbd_quilt_offscreen_depth.py:
      - <base>.meta.json
      - <base>_center_depth.png
    where <base> is the quilt filename without extension (including _qsCxRaAR suffix).
    """
    base_noext = path.with_suffix("")
    meta = base_noext.with_suffix(".meta.json")
    depth = base_noext.parent / f"{base_noext.name}_center_depth.png"
    if not meta.exists():
        meta = None
    if not depth.exists():
        depth = None
    return meta, depth

@stub.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=FUNC_TIMEOUT,
    volumes={"/vol/weights": weights_vol, "/vol/outputs": outputs_vol},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
)
def gen3c_render_from_quilt(
    quilt_image_bytes: bytes,
    orig_filename: str,
    base_name: str,
    guidance: float,
    use_offload: bool,
    meta_json_bytes: Optional[bytes] = None,
    center_depth_bytes: Optional[bytes] = None,
) -> Dict[str, str]:
    _log(f"Worker started for {base_name} [file={orig_filename}]")
    _git_pull_repo()
    _ensure_checkpoints("HUGGINGFACE_TOKEN")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # Fallback to filename parsing if meta missing
    cols, rows = _parse_quilt_meta_from_name(orig_filename)

    tmp_img = _write_temp_png(quilt_image_bytes)
    tmp_meta: Optional[str] = None
    tmp_depth: Optional[str] = None
    try:
        if meta_json_bytes:
            tmp_meta = _write_temp_bytes(meta_json_bytes, suffix=".json")
            # If the meta includes cols/rows, it's fine; we still pass explicit grid/order.
        if center_depth_bytes:
            tmp_depth = _write_temp_bytes(center_depth_bytes, suffix=".png")

        env = {"PYTHONPATH": REPO_DIR}
        conditioning_name = f"{base_name}_conditioning_from_quilt.mp4"
        cmd = [
            "python", f"{REPO_DIR}/cosmos_predict1/diffusion/inference/gen3c_from_quilt.py",
            "--quilt_path", tmp_img,
            "--grid_cols", str(cols),
            "--grid_rows", str(rows),
            "--order", "quilt",
            "--checkpoint_dir", WEIGHTS_DIR,
            "--video_save_folder", OUTPUTS_DIR,
            "--video_save_name", base_name,
            "--guidance", str(guidance),
            "--num_steps", "25",
            "--seed", "1234",
            "--fps", "24",
            "--width", str(TARGET_W),
            "--height", str(TARGET_H),
            "--disable_guardrail",
            "--disable_prompt_upsampler",
            "--disable_prompt_encoder",
            "--strict_exact_length",
            "--save_conditioning_video",
            "--conditioning_video_name", conditioning_name,
        ]
        # NEW: pass sidecars to the new interface
        if tmp_meta:
            cmd += ["--meta_json", tmp_meta]
        if tmp_depth:
            cmd += ["--center_depth_path", tmp_depth]

        if use_offload:
            cmd += ["--offload_diffusion_transformer", "--offload_tokenizer", "--offload_text_encoder_model", "--offload_prompt_upsampler", "--offload_guardrail_models"]

        _log(f"BEGIN render from quilt: cols={cols}, rows={rows}, order=quilt, W={TARGET_W}, H={TARGET_H}")
        _run(cmd, cwd=REPO_DIR, extra_env=env)
        final_mp4 = os.path.join(OUTPUTS_DIR, f"{base_name}.mp4")
        conditioning_mp4 = os.path.join(OUTPUTS_DIR, conditioning_name)
        if not os.path.exists(final_mp4):
            raise FileNotFoundError(f"Expected output missing: {final_mp4}")
        outputs_vol.commit()
        _log(f"Done: {final_mp4}")
        return {
            "final": os.path.basename(final_mp4),
            "conditioning": os.path.basename(conditioning_mp4) if os.path.exists(conditioning_mp4) else ""
        }
    finally:
        try:
            os.remove(tmp_img)
        except OSError:
            pass
        if tmp_meta:
            try: os.remove(tmp_meta)
            except OSError: pass
        if tmp_depth:
            try: os.remove(tmp_depth)
            except OSError: pass

@stub.function(
    image=image,
    gpu=None,
    timeout=60 * 60,
    volumes={"/vol/outputs": outputs_vol, "/vol/weights": weights_vol},
)
def process_one_quilt(
    quilt_image_bytes: bytes,
    orig_filename: str,
    base_name: str,
    guidance: float,
    use_offload: bool,
    meta_json_bytes: Optional[bytes] = None,
    center_depth_bytes: Optional[bytes] = None,
) -> Dict[str, str]:
    call = gen3c_render_from_quilt.spawn(
        quilt_image_bytes=quilt_image_bytes,
        orig_filename=orig_filename,
        base_name=base_name,
        guidance=guidance,
        use_offload=use_offload,
        meta_json_bytes=meta_json_bytes,
        center_depth_bytes=center_depth_bytes,
    )
    return call.get()

@stub.local_entrypoint()
def main(
    input_image_path: str = "",
    output_dir: str = "output",
    guidance: float = 1.0,
    movement_distance: float = 0.25,  # ignored for quilts
    offload: int = 0,
    max_concurrent_images: int = 1,
    stop_on_error: int = 1,  # <-- fail fast by default
) -> None:
    if not input_image_path:
        input_image_path = "input_quilts"
    paths = _gather_quilts(input_image_path)
    if not paths:
        print(f"No quilt images found at: {input_image_path}", file=sys.stderr)
        sys.exit(2)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{_now()}] Found {len(paths)} quilt image(s) under {input_image_path}")
    print(f"[{_now()}] Note: movement_distance is ignored for quilts; strict_exact_length enforced to match model chunk size.")

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
    had_error = False

    for p in paths:
        if had_error:
            break
        base = p.stem
        img_bytes = p.read_bytes()

        # NEW: try to read sidecars (optional)
        meta_path, depth_path = _find_sidecars_for_quilt(p)
        meta_bytes = meta_path.read_bytes() if (meta_path and meta_path.exists()) else None
        depth_bytes = depth_path.read_bytes() if (depth_path and depth_path.exists()) else None

        print(f"[{_now()}] Submitting: {p} (base={base}) "
              f"{'[+meta]' if meta_bytes else ''}{'[+depth]' if depth_bytes else ''}")

        call = process_one_quilt.spawn(
            quilt_image_bytes=img_bytes,
            orig_filename=p.name,
            base_name=base,
            guidance=guidance,
            use_offload=bool(offload),
            meta_json_bytes=meta_bytes,
            center_depth_bytes=depth_bytes,
        )
        active.append((call, base, img_bytes))
        if len(active) >= max_concurrent_images:
            call0, base0, _ = active.pop(0)
            try:
                manifest = call0.get()
            except Exception as e:
                print(f"[{_now()}] ERROR: quilt '{base0}' failed: {e}", file=sys.stderr)
                had_error = True
                if stop_on_error:
                    sys.exit(1)
                continue
            final = manifest.get("final") or ""
            conditioning = manifest.get("conditioning") or ""
            if final:
                _fetch(final, out_dir / final, "final video")
                if (out_dir / final).name != f"{base0}.mp4":
                    try:
                        shutil.copy2(out_dir / final, out_dir / f"{base0}.mp4")
                        print(f"[{_now()}] Saved convenience copy: {out_dir / f'{base0}.mp4'}")
                    except Exception:
                        pass
            else:
                print(f"[{_now()}] ERROR: no final video for '{base0}'", file=sys.stderr)
                had_error = True
                if stop_on_error:
                    sys.exit(1)
            if conditioning:
                _fetch(conditioning, out_dir / conditioning, "conditioning video")

    # drain remaining
    while active and not had_error:
        call, base, _ = active.pop(0)
        try:
            manifest = call.get()
        except Exception as e:
            print(f"[{_now()}] ERROR: quilt '{base}' failed: {e}", file=sys.stderr)
            if stop_on_error:
                sys.exit(1)
            had_error = True
            continue
        final = manifest.get("final") or ""
        conditioning = manifest.get("conditioning") or ""
        if final:
            _fetch(final, out_dir / final, "final video")
            if (out_dir / final).name != f"{base}.mp4":
                try:
                    shutil.copy2(out_dir / final, out_dir / f"{base}.mp4")
                    print(f"[{_now()}] Saved convenience copy: {out_dir / f'{base}.mp4'}")
                except Exception:
                    pass
        else:
            print(f"[{_now()}] ERROR: no final video for '{base}'", file=sys.stderr)
            if stop_on_error:
                sys.exit(1)
            had_error = True
        if conditioning:
            _fetch(conditioning, out_dir / conditioning, "conditioning video")

    if had_error:
        print(f"[{_now()}] Done with errors.", file=sys.stderr)
        if stop_on_error:
            sys.exit(1)
    else:
        print(f"[{_now()}] Done.")
