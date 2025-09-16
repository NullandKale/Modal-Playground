# LKGCamera.py
import math, numpy as np
from typing import Tuple

class LKGCamera:
    def __init__(self, size: float, fov: float, viewcone: float, aspect: float, near: float, far: float):
        self.size = float(size); self.fov = float(fov); self.viewcone = float(viewcone)
        self.aspect = float(aspect); self.near = float(near); self.far = float(far)

    def _camera_distance(self) -> float:
        return self.size / math.tan(math.radians(self.fov))

    def _camera_offset(self) -> float:
        return self._camera_distance() * math.tan(math.radians(self.viewcone))

    def _view_matrix(self, offset: float, invert: bool) -> np.ndarray:
        f = np.array([0.0,0.0,1.0], dtype=np.float32); u = np.array([0.0,(-1.0 if invert else 1.0),0.0], dtype=np.float32)
        s = np.cross(f,u); s /= np.linalg.norm(s); u = np.cross(s,f)
        m = np.eye(4, dtype=np.float32)
        m[0,0],m[1,0],m[2,0] = s; m[0,1],m[1,1],m[2,1] = u; m[0,2],m[1,2],m[2,2] = -f
        m[0,3] = offset; m[2,3] = -self._camera_distance()
        if not invert: m = np.diag([-1.0,1.0,1.0,1.0]).astype(np.float32) @ m
        return m

    def _projection_matrix(self) -> np.ndarray:
        f = 1.0 / math.tan(math.radians(self.fov) * 0.5); n, fp = self.near, self.far
        m = np.zeros((4,4), dtype=np.float32); m[0,0] = f / self.aspect; m[1,1] = f
        m[2,2] = (fp + n) / (n - fp); m[2,3] = (2 * fp * n) / (n - fp); m[3,2] = -1.0
        return m

    def view_proj(self, normalized_view: float, invert: bool, depthiness: float, focus: float) -> Tuple[np.ndarray,np.ndarray]:
        vfc = normalized_view - 0.5; offset = -(vfc) * depthiness * self._camera_offset()
        view = self._view_matrix(offset, invert); proj = self._projection_matrix()
        if invert: focus = -focus
        proj[0,2] += (offset * 2.0 / (self.size * self.aspect)) + (vfc * focus)
        return view, proj
