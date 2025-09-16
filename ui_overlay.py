# ui_overlay.py
import os
import platform
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gl_utils import create_texture_rgba, update_texture_rgba


def _load_font(pt_size: int) -> ImageFont.FreeTypeFont:
    if pt_size < 8:
        pt_size = 8
    candidates: List[str] = []
    if platform.system() == "Windows":
        winfonts = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
        candidates += [
            os.path.join(winfonts, "segoeui.ttf"),
            os.path.join(winfonts, "arial.ttf"),
            os.path.join(winfonts, "verdana.ttf"),
            os.path.join(winfonts, "tahoma.ttf")
        ]
    elif platform.system() == "Darwin":
        candidates += [
            "/System/Library/Fonts/SFNS.ttf",
            "/System/Library/Fonts/SFNSDisplay.ttf",
            "/Library/Fonts/Arial.ttf",
            "/Library/Fonts/HelveticaNeue.dfont"
        ]
    else:
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
        ]
    for p in candidates:
        try:
            if os.path.exists(p):
                return ImageFont.truetype(p, pt_size)
        except Exception:
            pass
    try:
        return ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()


class UISlider:
    def __init__(self, key: str, label: str, vmin: float, vmax: float, vstep: float, value: float):
        self.key: str = key
        self.label: str = label
        self.vmin: float = vmin
        self.vmax: float = vmax
        self.vstep: float = vstep
        self.value: float = value
        self.rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self.bar: Tuple[int, int, int, int] = (0, 0, 0, 0)


class UIButton:
    def __init__(self, key: str, label: str):
        self.key: str = key
        self.label: str = label
        self.rect: Tuple[int, int, int, int] = (0, 0, 0, 0)


class UICheck:
    def __init__(self, key: str, label: str, value: bool):
        self.key: str = key
        self.label: str = label
        self.value: bool = value
        self.rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self.box: Tuple[int, int, int, int] = (0, 0, 0, 0)


class UIPanel:
    def __init__(self, width: int = 360, font_points: int = 20):
        self.width: int = width
        self.font: ImageFont.FreeTypeFont = _load_font(font_points)
        try:
            asc, desc = self.font.getmetrics()
            self.line_height: int = asc + desc
        except Exception:
            self.line_height = 18
        self.bg: Tuple[int, int, int, int] = (0, 0, 0, 200)
        self.fg: Tuple[int, int, int, int] = (240, 240, 240, 255)
        self.ac: Tuple[int, int, int, int] = (80, 160, 255, 255)
        self.line: Tuple[int, int, int, int] = (255, 255, 255, 64)
        self.slider_bg: Tuple[int, int, int, int] = (255, 255, 255, 40)
        self.slider_knob: Tuple[int, int, int, int] = (255, 255, 255, 200)
        self.btn_bg: Tuple[int, int, int, int] = (255, 255, 255, 28)
        self.check_bg: Tuple[int, int, int, int] = (255, 255, 255, 28)
        self.sliders: List[UISlider] = []
        self.buttons: List[UIButton] = []
        self.checks: List[UICheck] = []
        self.img_w: int = self.width
        self.img_h: int = 1
        self.texture: Optional[int] = None
        self.active_slider: Optional[str] = None

    def ensure_texture(self) -> int:
        if self.texture is None:
            self.texture = create_texture_rgba(np.zeros((max(1, self.img_h), self.img_w, 4), dtype=np.uint8), True)
        return self.texture

    def layout(self, height: int) -> None:
        self.img_h = height

    def _measure(self, d: ImageDraw.ImageDraw, text: str) -> Tuple[int, int]:
        try:
            bbox = d.textbbox((0, 0), text, font=self.font)
            return (bbox[2] - bbox[0], bbox[3] - bbox[1])
        except Exception:
            try:
                return self.font.getsize(text)
            except Exception:
                try:
                    w = int(d.textlength(text, font=self.font))
                except Exception:
                    w = max(1, 8 * len(text))
                try:
                    asc, desc = self.font.getmetrics()
                    h = asc + desc
                except Exception:
                    h = 12
                return (w, h)

    def draw(self, state: Dict[str, Any], mouse: Tuple[int, int], mouse_down: bool) -> np.ndarray:
        W: int = self.img_w
        H: int = self.img_h
        img = Image.new("RGBA", (W, H), self.bg)
        d = ImageDraw.Draw(img)
        y: int = 10
        d.text((10, y), "RGBD Quilt UI", font=self.font, fill=self.fg)
        y += self.line_height + 2
        d.line((10, y, W - 10, y), fill=self.line)
        y += 10
        y = self._draw_image(d, y, state)
        y = self._draw_camera(d, y, state)
        y = self._draw_mesh(d, y, state)
        y = self._draw_discard(d, y, state)
        y = self._draw_actions(d, y, state)
        if mouse_down and self.active_slider is not None:
            s = next((s for s in self.sliders if s.key == self.active_slider), None)
            if s is not None:
                bx0, by0, bx1, by1 = s.bar
                t = 0.0 if bx1 == bx0 else (mouse[0] - bx0) / float(bx1 - bx0)
                t = max(0.0, min(1.0, t))
                val = s.vmin + t * (s.vmax - s.vmin)
                if s.vstep > 0.0:
                    val = round(val / s.vstep) * s.vstep
                state[s.key] = float(val)
                s.value = float(val)
        return np.asarray(img, dtype=np.uint8)

    def _slider(self, d: ImageDraw.ImageDraw, y: int, s: UISlider, W: int) -> int:
        d.text((10, y), s.label, font=self.font, fill=self.fg)
        val_str = f"{s.value:.4f}" if (s.vmax - s.vmin) < 5.0 else f"{s.value:.2f}"
        d.text((W - 10 - 80, y), val_str, font=self.font, fill=self.fg)
        y += self.line_height
        x0: int = 16
        x1: int = W - 16
        y0: int = y
        y1: int = y + 10
        d.rectangle((x0, y0, x1, y1), fill=self.slider_bg, outline=None)
        t = 0.0 if s.vmax == s.vmin else (s.value - s.vmin) / (s.vmax - s.vmin)
        cx: int = int(x0 + t * (x1 - x0))
        d.rectangle((cx - 4, y0 - 3, cx + 4, y1 + 3), fill=self.slider_knob, outline=self.ac)
        s.rect = (10, y - self.line_height, W - 10, y1 + 3)
        s.bar = (x0, y0, x1, y1)
        y += 18
        return y

    def _button_row(self, d: ImageDraw.ImageDraw, y: int, buttons: List[UIButton], W: int, cols: int) -> int:
        pad: int = 10
        bw: int = (W - pad * (cols + 1)) // cols
        _, sh = self._measure(d, "Sample")
        bh: int = max(30, sh + 12)
        for i, btn in enumerate(buttons):
            col: int = i % cols
            row: int = i // cols
            x0: int = pad + col * (bw + pad)
            x1: int = x0 + bw
            y0: int = y + row * (bh + 8)
            y1: int = y0 + bh
            btn.rect = (x0, y0, x1, y1)
            d.rectangle(btn.rect, fill=self.btn_bg, outline=self.line)
            tw, th = self._measure(d, btn.label)
            d.text((x0 + (bw - tw) // 2, y0 + (bh - th) // 2), btn.label, font=self.font, fill=self.fg)
        y += ((len(buttons) + cols - 1) // cols) * (bh + 8)
        y += 6
        return y

    def _checkbox(self, d: ImageDraw.ImageDraw, y: int, c: UICheck, W: int) -> int:
        x0: int = 16
        y0: int = y
        box = (x0, y0, x0 + 18, y0 + 18)
        d.rectangle(box, fill=self.check_bg, outline=self.line)
        if c.value:
            d.line((x0 + 3, y0 + 9, x0 + 8, y0 + 14), fill=self.fg, width=2)
            d.line((x0 + 8, y0 + 14, x0 + 15, y0 + 5), fill=self.fg, width=2)
        d.text((x0 + 24, y0), c.label, font=self.font, fill=self.fg)
        tw, th = self._measure(d, c.label)
        c.rect = (x0, y0, x0 + 18 + 6 + tw, y0 + max(18, th))
        c.box = box
        y += max(24, th + 6)
        return y

    def _draw_image(self, d: ImageDraw.ImageDraw, y: int, st: Dict[str, Any]) -> int:
        d.text((10, y), "Image", font=self.font, fill=self.ac)
        y += self.line_height
        if not any(s.key == "image_index" for s in self.sliders):
            self.sliders.insert(0, UISlider("image_index", "Index", 0.0, max(0.0, float(st["image_count"] - 1)), 1.0, float(st["image_index"])))
        else:
            s0 = next(s for s in self.sliders if s.key == "image_index")
            s0.vmax = max(0.0, float(st["image_count"] - 1))
            s0.value = float(st["image_index"])
        if not any(b.key == "prev" for b in self.buttons):
            self.buttons.append(UIButton("prev", "< Prev"))
        if not any(b.key == "next" for b in self.buttons):
            self.buttons.append(UIButton("next", "Next >"))
        s = next(s for s in self.sliders if s.key == "image_index")
        y = self._slider(d, y, s, self.img_w)
        y = self._button_row(d, y, [b for b in self.buttons if b.key in ("prev", "next")], self.img_w, cols=2)
        return y

    def _draw_camera(self, d: ImageDraw.ImageDraw, y: int, st: Dict[str, Any]) -> int:
        d.text((10, y), "Camera", font=self.font, fill=self.ac)
        y += self.line_height
        defs: List[Tuple[str, str, float, float, float, float]] = [
            ("depthiness", "Depthiness", 0.0, 3.0, 0.01, st["depthiness"]),
            ("focus", "Focus", -1.0, 1.0, 0.001, st["focus"]),
            ("camera_size", "Camera Size", 0.1, 6.0, 0.01, st["camera_size"]),
            ("fov_deg", "FOV (deg)", 5.0, 60.0, 0.1, st["fov_deg"]),
            ("viewcone_deg", "Viewcone (deg)", 5.0, 60.0, 0.1, st["viewcone_deg"])
        ]
        y = self._ensure_and_draw_sliders(d, y, defs)
        if not any(c.key == "invert_quilt" for c in self.checks):
            self.checks.append(UICheck("invert_quilt", "Invert quilt", bool(st["invert_quilt"])))
        else:
            c0 = next(c for c in self.checks if c.key == "invert_quilt")
            c0.value = bool(st["invert_quilt"])
        c = next(c for c in self.checks if c.key == "invert_quilt")
        y = self._checkbox(d, y, c, self.img_w)
        return y

    def _draw_mesh(self, d: ImageDraw.ImageDraw, y: int, st: Dict[str, Any]) -> int:
        d.text((10, y), "Mesh & Pose", font=self.font, fill=self.ac)
        y += self.line_height
        defs: List[Tuple[str, str, float, float, float, float]] = [
            ("mesh_scale", "Mesh Scale", 0.1, 3.0, 0.001, st["mesh_scale"]),
            ("yaw_deg", "Yaw (deg)", -45.0, 45.0, 0.1, st["yaw_deg"]),
            ("pitch_deg", "Pitch (deg)", -45.0, 45.0, 0.1, st["pitch_deg"]),
            ("roll_deg", "Roll (deg)", -45.0, 45.0, 0.1, st["roll_deg"]),
            ("preview_scale", "Preview Scale", 0.2, 1.0, 0.01, st["preview_scale"])
        ]
        y = self._ensure_and_draw_sliders(d, y, defs)
        return y

    def _draw_discard(self, d: ImageDraw.ImageDraw, y: int, st: Dict[str, Any]) -> int:
        d.text((10, y), "Discard Heuristics", font=self.font, fill=self.ac)
        y += self.line_height
        defs: List[Tuple[str, str, float, float, float, float]] = [
            ("radius", "Radius", 1.0, float(st["max_radius"]), 1.0, float(st["radius"])),
            ("step_uv_x", "Step UV X", 0.005, 0.1, 0.001, st["step_uv_x"]),
            ("step_uv_y", "Step UV Y", 0.005, 0.1, 0.001, st["step_uv_y"]),
            ("discard_range", "Discard Range", 0.01, 0.5, 0.001, st["discard_range"]),
            ("near_eps", "Near Eps", 0.001, 0.2, 0.001, st["near_eps"]),
            ("min_near_frac", "Min Near Frac", 0.0, 1.0, 0.01, st["min_near_frac"])
        ]
        y = self._ensure_and_draw_sliders(d, y, defs)
        return y

    def _draw_actions(self, d: ImageDraw.ImageDraw, y: int, st: Dict[str, Any]) -> int:
        d.text((10, y), "Actions", font=self.font, fill=self.ac)
        y += self.line_height
        for k, label in [
            ("render_preview", "Render Preview"),
            ("save_png", "Save PNG"),
            ("save_json", "Save Config"),
            ("recompute_depth", "Recompute Depth"),
            ("reset_defaults", "Reset Defaults")
        ]:
            if not any(b.key == k for b in self.buttons):
                self.buttons.append(UIButton(k, label))
        y = self._button_row(d, y, [b for b in self.buttons if b.key in ("render_preview", "save_png")], self.img_w, cols=2)
        y = self._button_row(d, y, [b for b in self.buttons if b.key in ("save_json", "recompute_depth")], self.img_w, cols=2)
        y = self._button_row(d, y, [b for b in self.buttons if b.key in ("reset_defaults",)], self.img_w, cols=1)
        return y

    def _ensure_and_draw_sliders(self, d: ImageDraw.ImageDraw, y: int, defs: List[Tuple[str, str, float, float, float, float]]) -> int:
        for key, label, vmin, vmax, vstep, val in defs:
            slot = next((s for s in self.sliders if s.key == key), None)
            if slot is None:
                self.sliders.append(UISlider(key, label, float(vmin), float(vmax), float(vstep), float(val)))
            else:
                slot.vmin = float(vmin)
                slot.vmax = float(vmax)
                slot.vstep = float(vstep)
                slot.value = float(val)
        for key, _, _, _, _, _ in defs:
            s = next(s for s in self.sliders if s.key == key)
            y = self._slider(d, y, s, self.img_w)
        return y

    def handle_mouse_down(self, x: int, y: int) -> Optional[str]:
        for s in self.sliders:
            x0, y0, x1, y1 = s.bar
            if x0 <= x <= x1 and (y0 - 10) <= y <= (y1 + 10):
                self.active_slider = s.key
                return "slider"
        for b in self.buttons:
            x0, y0, x1, y1 = b.rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                return f"button:{b.key}"
        for c in self.checks:
            x0, y0, x1, y1 = c.rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                c.value = not c.value
                return f"check:{c.key}"
        return None

    def clear_active(self) -> None:
        self.active_slider = None
