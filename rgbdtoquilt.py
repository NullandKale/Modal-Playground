#!/usr/bin/env python3
# rgbd_quilt_offscreen_depth.py
# Off-screen RGB→Depth→Quilt OpenGL renderer (HQ JPEG) using QuiltRenderer.
# Now also writes:
#   - <quilt>.meta.json   (camera + quilt metadata)
#   - <quilt>_center_depth.png  (16-bit PNG, 1.0=NEAR)

import os
import sys
import math
import ctypes
import time
import argparse
import platform
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
import glfw
from OpenGL import GL

# NEW: use the shared renderer class
from renderer import QuiltRenderer

# ============================= CONFIG (single place) =============================
CONFIG = dict(
    # Quilt grid & output
    COLS=11, ROWS=11,
    JPEG_QUALITY=85,             # set 95–98 and subsampling=0 if you want near-lossless

    # Camera / LKG parameters
    FOV_DEG=14.0,
    VIEWCONE_DEG=40.0,
    NEAR=0.1,
    FAR=100.0,
    DEPTHINESS=1.0,
    FOCUS=0.0,
    CAMERA_SIZE=2.0,             # volume size → parallax
    INVERT_QUILT=True,           # invert per-view Y in camera (typical for quilts)

    # Mesh (tessellated plane extruded by depth)
    MESH_SUBDIV=1024,            # (nx=ny) subdivisions; higher → smoother, heavier
    MESH_SCALE=1.0,              # uniform scale
    YAW_DEG=0.0, PITCH_DEG=0.0, ROLL_DEG=0.0,

    # Discard heuristics (all normalized, resolution-independent, pixel-space forbidden)
    MAX_RADIUS=12,               # hard cap for shader loop in renderer
    RADIUS=4,                    # runtime neighborhood half-width (<= MAX_RADIUS)
    STEP_UV_X=0.025,             # normalized UV half-size of sampling neighborhood in U
    STEP_UV_Y=0.025,             # normalized UV half-size of sampling neighborhood in V
    DISCARD_RANGE=0.08,          # (max-min) threshold across UV neighborhood (in [-0.5..0.5] depth)
    NEAR_EPS=0.040,              # tolerance to treat a sample as “near” wrt near ridge
    MIN_NEAR_FRAC=0.25,          # kept for compatibility with renderer API

    # Depth model
    DEPTH_MODEL="depth-anything/Depth-Anything-V2-Large-hf",  # HuggingFace ID
    DEPTH_DEVICE="cuda",          # "auto" | "cpu" | "cuda"

    # GL context attempts
    GL_ATTEMPTS=[(4,6)],
)

# ============================= utils: files =============================
def _gather_images(root: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    if root.is_file():
        return [root]
    if root.is_dir():
        return sorted([p for p in root.iterdir() if p.suffix.lower() in exts])
    return []

# ============================= GL texture helpers =============================
def _create_texture_rgba(img_rgba: np.ndarray) -> int:
    h, w, c = img_rgba.shape
    assert c == 4
    tex = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, w, h, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, img_rgba)
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    return tex

def _delete_texture(tex: Optional[int]) -> None:
    if tex:
        GL.glDeleteTextures(1, [tex])

# ============================= quilt FBO readback =============================
def _read_quilt_image(fbo: int, qw: int, qh: int) -> np.ndarray:
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
    GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
    data = GL.glReadPixels(0, 0, qw, qh, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
    arr = np.frombuffer(data, dtype=np.uint8).reshape((qh, qw, 4))
    return np.flipud(arr)

# ============================= intrinsics helper =============================
def _intrinsics_from_fov(tile_w: int, tile_h: int, fov_deg: float) -> Tuple[float,float,float,float]:
    f = 0.5 * tile_w / max(math.tan(math.radians(fov_deg) * 0.5), 1e-8)
    fx = fy = float(f)
    cx = (tile_w - 1) * 0.5
    cy = (tile_h - 1) * 0.5
    return fx, fy, cx, cy

# ============================= depth estimator =============================
class DepthEstimator:
    def __init__(self, model: str = CONFIG["DEPTH_MODEL"], device: str = CONFIG["DEPTH_DEVICE"]):
        self.model = model
        self.device = device
        self.pipe = None
        from transformers import pipeline
        self.pipe = pipeline(
            task="depth-estimation",
            model=self.model,
            device=(0 if (device == "cuda") else (-1 if device == "cpu" else (0 if _cuda_ok() else -1)))
        )

    def infer(self, pil_img: Image.Image) -> np.ndarray:
        out = self.pipe(pil_img)
        depth_img = out.get("depth") or out.get("predicted_depth")
        if not isinstance(depth_img, Image.Image):
            arr = np.array(depth_img, dtype=np.float32)
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            depth_img = Image.fromarray((arr * 255.0).astype(np.uint8))
        if depth_img.size != pil_img.size:
            depth_img = depth_img.resize(pil_img.size, Image.BICUBIC)
        return np.asarray(depth_img).astype(np.float32) / 255.0  # 1.0 = NEAR

def _cuda_ok() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

# ============================= GL context creation =============================
def create_hidden_gl_context(title: str = "offscreen") -> glfw._GLFWwindow:
    if not glfw.init():
        raise RuntimeError("GLFW init failed")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
    glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)
    is_macos = (platform.system() == "Darwin")
    last_err = None
    for (maj, minr) in CONFIG["GL_ATTEMPTS"]:
        try:
            glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_API)
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, maj)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, minr)
            if (maj > 3) or (maj == 3 and minr >= 2):
                glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            else:
                glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_ANY_PROFILE)
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE if is_macos else glfw.FALSE)
            glfw.window_hint(glfw.OPENGL_DEBUG_CONTEXT, glfw.TRUE)
            win = glfw.create_window(16, 16, f"{title} GL {maj}.{minr}", None, None)
            if not win:
                last_err = glfw.get_error()
                continue
            glfw.make_context_current(win)
            ver = GL.glGetString(GL.GL_VERSION)
            print(f"[GL] Context: {ver.decode() if hasattr(ver,'decode') else ver} (requested {maj}.{minr})")
            return win
        except Exception as e:
            last_err = e
            continue
    glfw.terminate()
    raise RuntimeError(f"Failed to create hidden GLFW window. Last error: {last_err}")

# ============================= main =============================
def main():
    parser = argparse.ArgumentParser(description="Off-screen RGB→Depth→Quilt OpenGL renderer (HQ JPEG) using QuiltRenderer.")
    # I/O
    parser.add_argument("--input_dir",  type=str, default="input")
    parser.add_argument("--output_dir", type=str, default="input_quilts")
    # Quilt grid
    parser.add_argument("--cols", type=int, default=CONFIG["COLS"])
    parser.add_argument("--rows", type=int, default=CONFIG["ROWS"])
    # Camera
    parser.add_argument("--fov_deg", type=float, default=CONFIG["FOV_DEG"])
    parser.add_argument("--viewcone_deg", type=float, default=CONFIG["VIEWCONE_DEG"])
    parser.add_argument("--near", type=float, default=CONFIG["NEAR"])
    parser.add_argument("--far", type=float, default=CONFIG["FAR"])
    parser.add_argument("--depthiness", type=float, default=CONFIG["DEPTHINESS"])
    parser.add_argument("--focus", type=float, default=CONFIG["FOCUS"])
    parser.add_argument("--camera_size", type=float, default=CONFIG["CAMERA_SIZE"])
    parser.add_argument("--invert_quilt", type=int, default=int(CONFIG["INVERT_QUILT"]))
    # Mesh
    parser.add_argument("--mesh_subdiv", type=int, default=CONFIG["MESH_SUBDIV"])
    parser.add_argument("--mesh_scale", type=float, default=CONFIG["MESH_SCALE"])
    parser.add_argument("--yaw_deg", type=float, default=CONFIG["YAW_DEG"])
    parser.add_argument("--pitch_deg", type=float, default=CONFIG["PITCH_DEG"])
    parser.add_argument("--roll_deg", type=float, default=CONFIG["ROLL_DEG"])
    # Discard (normalized UV)
    parser.add_argument("--radius", type=int, default=CONFIG["RADIUS"])
    parser.add_argument("--step_uv_x", type=float, default=CONFIG["STEP_UV_X"])
    parser.add_argument("--step_uv_y", type=float, default=CONFIG["STEP_UV_Y"])
    parser.add_argument("--discard_range", type=float, default=CONFIG["DISCARD_RANGE"])
    parser.add_argument("--near_eps", type=float, default=CONFIG["NEAR_EPS"])
    parser.add_argument("--min_near_frac", type=float, default=CONFIG["MIN_NEAR_FRAC"])
    # Depth
    parser.add_argument("--depth_model", type=str, default=CONFIG["DEPTH_MODEL"])
    parser.add_argument("--depth_device", type=str, choices=["auto","cpu","cuda"], default=CONFIG["DEPTH_DEVICE"])
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = _gather_images(in_dir)
    if not files:
        print(f"No images found in {in_dir}", file=sys.stderr)
        sys.exit(2)

    # Hidden GL context
    try:
        window = create_hidden_gl_context("offscreen")
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(3)

    try:
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_CULL_FACE)

        # Instantiate shared renderer and set discard parameters
        renderer = QuiltRenderer(CONFIG["MAX_RADIUS"], args.mesh_subdiv)
        renderer.set_discard_params(
            radius=int(args.radius),
            step_uv_x=float(args.step_uv_x),
            step_uv_y=float(args.step_uv_y),
            discard_range=float(args.discard_range),
            near_eps=float(args.near_eps),
            min_near_frac=float(args.min_near_frac),
            max_radius=int(CONFIG["MAX_RADIUS"])
        )

        # Depth estimator
        depth_est = DepthEstimator(model=args.depth_model, device=args.depth_device)

        for p in files:
            color = Image.open(p).convert("RGBA")
            W, H = color.size

            t0 = time.time()
            depth01 = depth_est.infer(color.convert("RGB"))  # 1.0 = NEAR
            t1 = time.time()

            # Pack combined RGBA (left=color, right=depth grayscale)
            color_rgba = np.asarray(color, dtype=np.uint8)              # HxWx4
            depth_u8   = np.clip((depth01 * 255.0 + 0.5).astype(np.uint8), 0, 255)
            depth_rgba = np.stack([depth_u8, depth_u8, depth_u8, np.full_like(depth_u8, 255)], axis=-1)
            combined   = np.concatenate([color_rgba, depth_rgba], axis=1)  # H x (2W) x 4

            atlas_tex = _create_texture_rgba(combined)

            aspect_color = float(W) / float(H)
            cols = int(args.cols)
            rows = int(args.rows)
            qw = cols * W
            qh = rows * H

            out_name = f"{p.stem}_qs{cols}x{rows}a{aspect_color:.4f}".rstrip('0').rstrip('.') + ".jpg"
            out_path = out_dir / out_name
            out_base = out_dir / out_name.rsplit(".", 1)[0]
            meta_path = out_base.with_suffix(".meta.json")
            depth16_path = out_base.parent / f"{out_base.name}_center_depth.png"

            print(f"[{time.strftime('%H:%M:%S')}] {p.name}: depth {H}x{W} in {(t1 - t0):.2f}s → quilt {qw}x{qh}")

            # Camera parameter block for renderer
            cam_params = {
                "camera_size": float(args.camera_size),
                "fov_deg": float(args.fov_deg),
                "viewcone_deg": float(args.viewcone_deg),
                "near": float(args.near),
                "far": float(args.far),
                "depthiness": float(args.depthiness),
                "focus": float(args.focus)
            }

            # Render quilt using the shared renderer
            fbo, quilt_tex, quilt_rb, qw, qh = renderer.render_quilt_to_fbo(
                combined_tex=atlas_tex,
                W=W,
                H=H,
                cols=cols,
                rows=rows,
                cam_params=cam_params,
                invert_quilt=bool(args.invert_quilt),
                mesh_scale=float(args.mesh_scale),
                yaw_deg=float(args.yaw_deg),
                pitch_deg=float(args.pitch_deg),
                roll_deg=float(args.roll_deg)
            )

            # Read back and save quilt
            arr = _read_quilt_image(fbo, qw, qh)
            Image.fromarray(arr).convert("RGB").save(
                out_path,
                "JPEG",
                quality=CONFIG["JPEG_QUALITY"],
                subsampling=0,
                optimize=True
            )
            print(f"[{time.strftime('%H:%M:%S')}] Saved quilt: {out_path}")

            # --------- NEW: write sidecars ---------

            # 16-bit center depth (1.0 = NEAR, white)
            depth_u16 = np.clip((depth01 * 65535.0 + 0.5).astype(np.uint16), 0, 65535)
            Image.fromarray(depth_u16, mode="I;16").save(depth16_path)
            print(f"[{time.strftime('%H:%M:%S')}] Saved center depth: {depth16_path}")

            # Metadata JSON
            fx, fy, cx, cy = _intrinsics_from_fov(W, H, float(args.fov_deg))
            center_rc = [rows // 2, cols // 2]  # symmetric center for odd grids

            meta = {
                "version": 1,
                "timestamp": datetime.now().isoformat(),
                # Quilt layout
                "cols": cols,
                "rows": rows,
                "order": "quilt",               # bottom-left origin, row-major (matches extractor)
                "tile_width": int(W),
                "tile_height": int(H),
                "quilt_width": int(qw),
                "quilt_height": int(qh),
                "aspect_color": float(aspect_color),
                "center_tile_rc": center_rc,    # [row, col] in grid space
                # Camera & projection
                "fov_deg": float(args.fov_deg),
                "viewcone_deg": float(args.viewcone_deg),
                "camera_size": float(args.camera_size),
                "invert_quilt": bool(args.invert_quilt),
                "near": float(args.near),
                "far": float(args.far),
                "depthiness": float(args.depthiness),
                "focus": float(args.focus),
                # Intrinsics at tile resolution (horizontal FOV)
                "intrinsics_px": {
                    "fx": float(fx), "fy": float(fy),
                    "cx": float(cx), "cy": float(cy)
                },
                # Renderer knobs (for reproducibility / debugging)
                "mesh": {
                    "subdiv": int(args.mesh_subdiv),
                    "scale": float(args.mesh_scale),
                    "yaw_deg": float(args.yaw_deg),
                    "pitch_deg": float(args.pitch_deg),
                    "roll_deg": float(args.roll_deg)
                },
                "discard": {
                    "max_radius": int(CONFIG["MAX_RADIUS"]),
                    "radius": int(args.radius),
                    "step_uv_x": float(args.step_uv_x),
                    "step_uv_y": float(args.step_uv_y),
                    "discard_range": float(args.discard_range),
                    "near_eps": float(args.near_eps),
                    "min_near_frac": float(args.min_near_frac)
                },
                # Depth side info
                "depth_model": str(args.depth_model),
                "depth_device": str(args.depth_device),
                "depth_white_is_near": True,
                # Sidecar filenames
                "quilt_filename": os.path.basename(out_path),
                "center_depth_filename": os.path.basename(depth16_path)
            }

            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            print(f"[{time.strftime('%H:%M:%S')}] Saved metadata: {meta_path}")

            # Cleanup per-image GL objects
            GL.glDeleteFramebuffers(1, [fbo])
            GL.glDeleteTextures(1, [quilt_tex])
            GL.glDeleteRenderbuffers(1, [quilt_rb])
            _delete_texture(atlas_tex)

        # Cleanup renderer and window
        renderer.cleanup()
        glfw.destroy_window(window)
    finally:
        glfw.terminate()

if __name__ == "__main__":
    main()
