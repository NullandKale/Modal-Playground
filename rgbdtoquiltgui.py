# app_main.py
#!/usr/bin/env python3
import os, sys, time, math, json, datetime, argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image
import glfw
from OpenGL import GL
from gl_utils import create_gl_context, make_fullscreen_present, present_texture, create_texture_rgba, update_texture_rgba, delete_texture, read_fbo_rgba
from ui_overlay import UIPanel
from renderer import QuiltRenderer
from depth_estimator import DepthEstimator, _cuda_ok

CONFIG = dict(COLS=11, ROWS=11, FOV_DEG=14.0, VIEWCONE_DEG=40.0, NEAR=0.1, FAR=100.0, DEPTHINESS=1.0, FOCUS=0.0, CAMERA_SIZE=2.0, INVERT_QUILT=True, MESH_SUBDIV=512, MESH_SCALE=1.0, YAW_DEG=0.0, PITCH_DEG=0.0, ROLL_DEG=0.0, MAX_RADIUS=12, RADIUS=4, STEP_UV_X=0.025, STEP_UV_Y=0.025, DISCARD_RANGE=0.08, NEAR_EPS=0.040, MIN_NEAR_FRAC=0.25, DEPTH_MODEL="depth-anything/Depth-Anything-V2-Large-hf", DEPTH_DEVICE="cuda")

def _gather_images(root: Path):
    exts={".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp"}
    if root.is_file(): return [root]
    if root.is_dir(): return sorted([p for p in root.iterdir() if p.suffix.lower() in exts])
    return []

class App:
    def __init__(self, args):
        self.args=args
        self.in_dir=Path(args.input_dir); self.out_dir=Path(args.output_dir); self.out_dir.mkdir(parents=True, exist_ok=True)
        self.files=_gather_images(self.in_dir)
        if not self.files: print(f"No images found in {self.in_dir}", file=sys.stderr); sys.exit(2)
        self.window=create_gl_context("rgbd-quilt-ui", True, 1600, 900, attempts=((4,6),))
        GL.glEnable(GL.GL_DEPTH_TEST); GL.glDisable(GL.GL_CULL_FACE); GL.glEnable(GL.GL_BLEND); GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        self.present_prog, self.present_loc_tex, self.present_loc_flip, self.present_vao, self.present_vbo, self.present_ebo, self.present_icount = make_fullscreen_present()
        self.renderer=QuiltRenderer(CONFIG["MAX_RADIUS"], args.mesh_subdiv)
        self.depth_est=DepthEstimator(model=args.depth_model, device=args.depth_device)
        self.ui=UIPanel(width=360, font_points=20); self.ui.ensure_texture()
        self.state=self._default_state()
        self.combined_cache: Dict[Path, Tuple[int,int,int]]={}
        self.preview_tex=None; self.preview_fbo=None; self.preview_rb=None; self.preview_w=0; self.preview_h=0
        self.mouse_down=False; self.request_preview=True; self.request_fullsave=False; self.request_jsonsave=False; self.recompute_depth=False
        self._bind_callbacks()

    def _default_state(self) -> Dict[str,Any]:
        return {"image_index":0.0,"image_count":len(self.files),"cols":float(CONFIG["COLS"]),"rows":float(CONFIG["ROWS"]),"fov_deg":float(CONFIG["FOV_DEG"]), "viewcone_deg":float(CONFIG["VIEWCONE_DEG"]), "near":float(CONFIG["NEAR"]), "far":float(CONFIG["FAR"]), "depthiness":float(CONFIG["DEPTHINESS"]), "focus":float(CONFIG["FOCUS"]), "camera_size":float(CONFIG["CAMERA_SIZE"]), "invert_quilt":bool(CONFIG["INVERT_QUILT"]), "mesh_scale":float(CONFIG["MESH_SCALE"]), "yaw_deg":float(CONFIG["YAW_DEG"]), "pitch_deg":float(CONFIG["PITCH_DEG"]), "roll_deg":float(CONFIG["ROLL_DEG"]), "max_radius":int(CONFIG["MAX_RADIUS"]), "radius":float(CONFIG["RADIUS"]), "step_uv_x":float(CONFIG["STEP_UV_X"]), "step_uv_y":float(CONFIG["STEP_UV_Y"]), "discard_range":float(CONFIG["DISCARD_RANGE"]), "near_eps":float(CONFIG["NEAR_EPS"]), "min_near_frac":float(CONFIG["MIN_NEAR_FRAC"]), "preview_scale":0.5}

    def _bind_callbacks(self) -> None:
        glfw.set_window_user_pointer(self.window, self)
        def on_key(win, key, scancode, action, mods):
            if action not in (glfw.PRESS, glfw.REPEAT): return
            if key == glfw.KEY_ESCAPE: glfw.set_window_should_close(win, True)
            elif key == glfw.KEY_LEFT: self._change_image(-1)
            elif key == glfw.KEY_RIGHT: self._change_image(+1)
            elif key == glfw.KEY_ENTER: self.request_fullsave = True
            elif key == glfw.KEY_C: self.request_jsonsave = True
            elif key == glfw.KEY_I: self.state["invert_quilt"] = not self.state["invert_quilt"]; self.request_preview = True
            elif key == glfw.KEY_R: self._reset_defaults()
        def on_mouse_button(win, button, action, mods):
            if button != glfw.MOUSE_BUTTON_LEFT: return
            x, y = glfw.get_cursor_pos(win); w, h = glfw.get_window_size(win); panel_x0 = w - self.ui.width
            if action == glfw.PRESS:
                if x >= panel_x0:
                    lx = int(x - panel_x0); ly = int(y)
                    hit = self.ui.handle_mouse_down(lx, ly)
                    if hit is not None:
                        if hit.startswith("button:"): self._handle_button(hit.split(":",1)[1])
                        elif hit.startswith("check:"):
                            key = hit.split(":",1)[1]
                            if key == "invert_quilt": self.state["invert_quilt"] = any(c.key=="invert_quilt" and c.value for c in self.ui.checks); self.request_preview = True
                        else: self.request_preview = True
                    self.mouse_down = True
            elif action == glfw.RELEASE:
                self.ui.clear_active(); self.mouse_down = False
        def on_cursor(win, x, y): pass
        glfw.set_key_callback(self.window, on_key); glfw.set_mouse_button_callback(self.window, on_mouse_button); glfw.set_cursor_pos_callback(self.window, on_cursor)

    def _handle_button(self, key: str) -> None:
        if key == "prev": self._change_image(-1)
        elif key == "next": self._change_image(+1)
        elif key == "render_preview": self.request_preview = True
        elif key == "save_png": self.request_fullsave = True
        elif key == "save_json": self.request_jsonsave = True
        elif key == "recompute_depth": self.recompute_depth = True; self.request_preview = True
        elif key == "reset_defaults": self._reset_defaults()

    def _reset_defaults(self) -> None:
        keep_idx = self.state.get("image_index", 0.0); keep_cnt = self.state.get("image_count", len(self.files))
        self.state = self._default_state()
        self.state["image_index"] = keep_idx
        self.state["image_count"] = keep_cnt
        self.request_preview = True

    def _change_image(self, delta: int) -> None:
        count = self.state["image_count"]; idx = int(self.state["image_index"]); idx = (idx + delta) % count
        self.state["image_index"] = float(idx); self.request_preview = True

    def _current_file(self) -> Path:
        return self.files[int(self.state["image_index"])]

    def _ensure_combined_texture(self, p: Path, force_recompute: bool = False) -> Tuple[int,int,int]:
        if (not force_recompute) and (p in self.combined_cache): return self.combined_cache[p]
        color = Image.open(p).convert("RGBA"); W, H = color.size
        t0 = time.time(); depth01 = self.depth_est.infer01(color.convert("RGB")); t1 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] {p.name}: depth {H}x{W} in {(t1-t0):.2f}s")
        color_rgba = np.asarray(color, dtype=np.uint8); depth_u8 = np.clip((depth01 * 255.0 + 0.5).astype(np.uint8), 0, 255)
        depth_rgba = np.stack([depth_u8, depth_u8, depth_u8, np.full_like(depth_u8, 255)], axis=-1)
        combined = np.concatenate([color_rgba, depth_rgba], axis=1)
        tex = create_texture_rgba(combined, True)
        self.combined_cache[p] = (tex, W, H)
        return tex, W, H

    def _render_preview(self) -> None:
        p = self._current_file(); tex, W, H = self._ensure_combined_texture(p, force_recompute=self.recompute_depth); self.recompute_depth = False
        scale = float(self.state["preview_scale"]); prevW = max(1, int(W * scale)); prevH = max(1, int(H * scale))
        self.renderer.set_discard_params(int(self.state["radius"]), float(self.state["step_uv_x"]), float(self.state["step_uv_y"]), float(self.state["discard_range"]), float(self.state["near_eps"]), float(self.state["min_near_frac"]), int(self.state["max_radius"]))
        cam_params = {"camera_size":float(self.state["camera_size"]), "fov_deg":float(self.state["fov_deg"]), "viewcone_deg":float(self.state["viewcone_deg"]), "near":float(self.state["near"]), "far":float(self.state["far"]), "depthiness":float(self.state["depthiness"]), "focus":float(self.state["focus"])}
        views = [0.0, 0.5, 1.0]
        fbo, quilt_tex, quilt_rb, qw, qh = self.renderer.render_selected_views_to_fbo(tex, prevW, prevH, views, cam_params, bool(self.state["invert_quilt"]), float(self.state["mesh_scale"]), float(self.state["yaw_deg"]), float(self.state["pitch_deg"]), float(self.state["roll_deg"]))
        if self.preview_fbo: GL.glDeleteFramebuffers(1, [self.preview_fbo])
        if self.preview_rb: GL.glDeleteRenderbuffers(1, [self.preview_rb])
        if self.preview_tex: GL.glDeleteTextures(1, [self.preview_tex])
        self.preview_fbo = fbo; self.preview_tex = quilt_tex; self.preview_rb = quilt_rb; self.preview_w = qw; self.preview_h = qh

    def _save_full_png(self) -> None:
        p = self._current_file(); tex, W, H = self._ensure_combined_texture(p, force_recompute=self.recompute_depth); self.recompute_depth = False
        cols = int(max(1, round(self.state["cols"]))); rows = int(max(1, round(self.state["rows"])))
        self.renderer.set_discard_params(int(self.state["radius"]), float(self.state["step_uv_x"]), float(self.state["step_uv_y"]), float(self.state["discard_range"]), float(self.state["near_eps"]), float(self.state["min_near_frac"]), int(self.state["max_radius"]))
        cam_params = {"camera_size":float(self.state["camera_size"]), "fov_deg":float(self.state["fov_deg"]), "viewcone_deg":float(self.state["viewcone_deg"]), "near":float(self.state["near"]), "far":float(self.state["far"]), "depthiness":float(self.state["depthiness"]), "focus":float(self.state["focus"])}
        fbo, quilt_tex, quilt_rb, qw, qh = self.renderer.render_quilt_to_fbo(tex, W, H, cols, rows, cam_params, bool(self.state["invert_quilt"]), float(self.state["mesh_scale"]), float(self.state["yaw_deg"]), float(self.state["pitch_deg"]), float(self.state["roll_deg"]))
        arr = read_fbo_rgba(fbo, qw, qh)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S"); out_path = self.out_dir / f"{p.stem}_qs{cols}x{rows}_{ts}.png"
        Image.fromarray(arr).save(out_path, "PNG", optimize=True); print(f"[{time.strftime('%H:%M:%S')}] Saved PNG: {out_path}")
        GL.glDeleteFramebuffers(1, [fbo]); GL.glDeleteTextures(1, [quilt_tex]); GL.glDeleteRenderbuffers(1, [quilt_rb])

    def _save_json(self) -> None:
        out = {"COLS": int(round(self.state["cols"])), "ROWS": int(round(self.state["rows"])), "FOV_DEG": float(self.state["fov_deg"]), "VIEWCONE_DEG": float(self.state["viewcone_deg"]), "NEAR": float(self.state["near"]), "FAR": float(self.state["far"]), "DEPTHINESS": float(self.state["depthiness"]), "FOCUS": float(self.state["focus"]), "CAMERA_SIZE": float(self.state["camera_size"]), "INVERT_QUILT": bool(self.state["invert_quilt"]), "MESH_SUBDIV": int(self.args.mesh_subdiv), "MESH_SCALE": float(self.state["mesh_scale"]), "YAW_DEG": float(self.state["yaw_deg"]), "PITCH_DEG": float(self.state["pitch_deg"]), "ROLL_DEG": float(self.state["roll_deg"]), "MAX_RADIUS": int(self.state["max_radius"]), "RADIUS": int(round(self.state["radius"])), "STEP_UV_X": float(self.state["step_uv_x"]), "STEP_UV_Y": float(self.state["step_uv_y"]), "DISCARD_RANGE": float(self.state["discard_range"]), "NEAR_EPS": float(self.state["near_eps"]), "MIN_NEAR_FRAC": float(self.state["min_near_frac"]), "DEPTH_MODEL": str(self.args.depth_model), "DEPTH_DEVICE": str(self.args.depth_device)}
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S"); out_path = self.out_dir / f"config_{ts}.json"
        with open(out_path, "w", encoding="utf-8") as f: json.dump(out, f, indent=2)
        print(f"[{time.strftime('%H:%M:%S')}] Saved config JSON: {out_path}")

    def _draw(self) -> None:
        ww, wh = glfw.get_window_size(self.window); panel_w = self.ui.width; view_w = max(1, ww - panel_w); view_h = wh
        GL.glViewport(0, 0, ww, wh); GL.glClearColor(0.05, 0.05, 0.06, 1.0); GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        if self.preview_tex: GL.glViewport(0, 0, view_w, view_h); present_texture(self.present_prog, self.present_loc_tex, self.present_loc_flip, self.present_vao, self.present_icount, self.preview_tex, flip_y=True)
        self.ui.layout(wh)
        mx, my = glfw.get_cursor_pos(self.window); panel_x0 = ww - panel_w
        overlay = self.ui.draw(self.state.copy(), (int(mx - panel_x0), int(my)), self.mouse_down and (mx >= panel_x0))
        self.ui.img_w = panel_w; self.ui.img_h = overlay.shape[0]
        if self.ui.texture is None: self.ui.texture = create_texture_rgba(overlay, True)
        else: update_texture_rgba(self.ui.texture, overlay, True)
        GL.glViewport(panel_x0, 0, panel_w, wh); present_texture(self.present_prog, self.present_loc_tex, self.present_loc_flip, self.present_vao, self.present_icount, self.ui.texture, flip_y=True)

    def _update_state_from_ui(self) -> None:
        changed = False
        for s in self.ui.sliders:
            val = float(s.value)
            if abs(self.state.get(s.key, val) - val) > 1e-12: self.state[s.key] = val; changed = True
        for c in self.ui.checks:
            if self.state.get(c.key, c.value) != c.value: self.state[c.key] = bool(c.value); changed = True
        if changed: self.request_preview = True

    def loop(self) -> None:
        try:
            while not glfw.window_should_close(self.window):
                glfw.poll_events()
                self._update_state_from_ui()
                if self.request_preview:
                    self.renderer.set_discard_params(int(self.state["radius"]), float(self.state["step_uv_x"]), float(self.state["step_uv_y"]), float(self.state["discard_range"]), float(self.state["near_eps"]), float(self.state["min_near_frac"]), int(self.state["max_radius"]))
                    self._render_preview(); self.request_preview = False
                if self.request_fullsave: self._save_full_png(); self.request_fullsave = False
                if self.request_jsonsave: self._save_json(); self.request_jsonsave = False
                self._draw(); glfw.swap_buffers(self.window)
        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        if self.preview_fbo: GL.glDeleteFramebuffers(1, [self.preview_fbo])
        if self.preview_rb: GL.glDeleteRenderbuffers(1, [self.preview_rb])
        if self.preview_tex: GL.glDeleteTextures(1, [self.preview_tex])
        for tex,_,_ in list(self.combined_cache.values()): delete_texture(tex)
        if self.ui.texture: GL.glDeleteTextures(1, [self.ui.texture])
        self.renderer.cleanup()
        GL.glDeleteProgram(self.present_prog); GL.glDeleteVertexArrays(1, [self.present_vao]); GL.glDeleteBuffers(1, [self.present_vbo]); GL.glDeleteBuffers(1, [self.present_ebo])
        glfw.destroy_window(self.window); glfw.terminate()

def main():
    parser = argparse.ArgumentParser(description="Interactive RGB→Depth→Quilt renderer with UI (PNG + JSON).")
    parser.add_argument("--input_dir",  type=str, default="input")
    parser.add_argument("--output_dir", type=str, default="output_quilts")
    parser.add_argument("--mesh_subdiv", type=int, default=CONFIG["MESH_SUBDIV"])
    parser.add_argument("--depth_model", type=str, default=CONFIG["DEPTH_MODEL"])
    parser.add_argument("--depth_device", type=str, choices=["auto","cpu","cuda"], default=CONFIG["DEPTH_DEVICE"])
    args = parser.parse_args()
    if args.depth_device == "auto": args.depth_device = "cuda" if _cuda_ok() else "cpu"
    app = App(args); app.loop()

if __name__ == "__main__":
    main()
