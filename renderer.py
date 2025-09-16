# renderer.py
# Quilt renderer: builds the tessellated plane, compiles shaders, and renders either a full quilt
# or a compact preview of selected views. This version adds an oriented, gradient-guided one-sided
# discard that avoids flat/stretchy regions and keeps only true occlusion cutouts.

import math
import ctypes
from typing import Tuple, List, Dict

import numpy as np
from OpenGL import GL

from gl_utils import link_program, init_quilt_fbo
from LKGCamera import LKGCamera


# ============================= Matrix helpers =============================

def _rot_y(a: float) -> np.ndarray:
    c = math.cos(a)
    s = math.sin(a)
    return np.array([
        [ c, 0.0,  s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s, 0.0,  c, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)


def _rot_x(a: float) -> np.ndarray:
    c = math.cos(a)
    s = math.sin(a)
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0,   c,  -s, 0.0],
        [0.0,   s,   c, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)


def _rot_z(a: float) -> np.ndarray:
    c = math.cos(a)
    s = math.sin(a)
    return np.array([
        [  c,  -s, 0.0, 0.0],
        [  s,   c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)


def _scale(sx: float, sy: float, sz: float) -> np.ndarray:
    return np.array([
        [ sx, 0.0, 0.0, 0.0],
        [0.0,  sy, 0.0, 0.0],
        [0.0, 0.0,  sz, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)


# ============================= Geometry =============================

def make_plane(nx: int, ny: int) -> Tuple[int, int, int]:
    xs = np.linspace(-1.0, 1.0, nx + 1, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, ny + 1, dtype=np.float32)

    positions: List[float] = []
    normals: List[float] = []
    uvs: List[float] = []
    indices: List[int] = []

    for j in range(ny + 1):
        v = j / float(ny)
        y = ys[j]
        for i in range(nx + 1):
            u = i / float(nx)
            x = xs[i]
            positions.extend([x, y, 0.0])
            normals.extend([0.0, 0.0, 1.0])
            uvs.extend([u, v])

    pos = np.array(positions, dtype=np.float32)
    nrm = np.array(normals, dtype=np.float32)
    uv = np.array(uvs, dtype=np.float32)

    row = nx + 1
    for j in range(ny):
        for i in range(nx):
            v0 = j * row + i
            v1 = v0 + 1
            v2 = v0 + row
            v3 = v2 + 1
            indices.extend([v0, v2, v1, v1, v2, v3])

    idx = np.array(indices, dtype=np.uint32)

    vao = GL.glGenVertexArrays(1)
    vbo_pos = GL.glGenBuffers(1)
    vbo_nrm = GL.glGenBuffers(1)
    vbo_uv = GL.glGenBuffers(1)
    ebo = GL.glGenBuffers(1)

    GL.glBindVertexArray(vao)

    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_pos)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, pos.nbytes, pos, GL.GL_STATIC_DRAW)
    GL.glEnableVertexAttribArray(0)
    GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_nrm)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, nrm.nbytes, nrm, GL.GL_STATIC_DRAW)
    GL.glEnableVertexAttribArray(1)
    GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_uv)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, uv.nbytes, uv, GL.GL_STATIC_DRAW)
    GL.glEnableVertexAttribArray(2)
    GL.glVertexAttribPointer(2, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ebo)
    GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL.GL_STATIC_DRAW)

    GL.glBindVertexArray(0)

    return vao, ebo, idx.size


# ============================= Shaders =============================

def make_shaders(max_radius: int) -> Tuple[str, str]:
    vert_src = r"""
    #version 330 core

    layout (location = 0) in vec3 position;
    layout (location = 1) in vec3 normal;
    layout (location = 2) in vec2 texCoords;

    out vec2 vColorUV;
    out vec2 vRawUV;
    out float vDepthVS;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    uniform sampler2D textureSampler;

    void main() {
        float d = 1.0 - texture(textureSampler, vec2(texCoords.x * 0.5 + 0.5, texCoords.y)).r;
        d -= 0.5;
        vDepthVS = d;

        vec3 displaced = position + normal * d;
        gl_Position = projection * view * model * vec4(displaced, 1.0);

        vColorUV = vec2(texCoords.x * 0.5, texCoords.y);
        vRawUV = texCoords;
    }
    """

    # Fragment shader:
    #  - Edge strength via central-difference (Sobel-lite) using a tiny step tied to uStepUV (uGradScale).
    #  - Oriented near/far scan along the depth gradient: find near-side ridge (max depth along +grad)
    #    and far-side floor (min depth along -grad), then discard only if center is truly behind the ridge
    #    and the near-far gap is large. This rejects flat/stretchy areas and keeps real occlusions.
    frag_template = r"""
    #version 330 core

    in vec2 vColorUV;
    in vec2 vRawUV;
    in float vDepthVS;

    out vec4 FragColor;

    uniform sampler2D textureSampler;

    // Legacy controls (kept for compatibility)
    uniform int   uRadius;
    uniform vec2  uStepUV;          // per-tap step in normalized UV (half-size / R)
    uniform float uDiscardRange;    // minimum near-far gap to consider an occlusion
    uniform float uNearEps;         // epsilon from near ridge to treat as background
    uniform float uMinNearFrac;     // retained (not required for new test but harmless)

    // New oriented-edge controls
    uniform float uGradScale;       // fraction of uStepUV used for gradient sampling (e.g., 0.5)
    uniform float uGradThresh;      // minimum gradient magnitude (in depth units) to be an edge
    uniform int   uMarchR;          // steps to march along +/- gradient (<= uRadius)

    const int MAX_RADIUS = $MAX_RADIUS;

    float sampleDepthSigned(vec2 rawUV) {
        float v = 1.0 - texture(textureSampler, vec2(rawUV.x * 0.5 + 0.5, rawUV.y)).r;
        return v - 0.5;
    }

    void main() {
        int R = uRadius;
        if (R > MAX_RADIUS) {
            R = MAX_RADIUS;
        }

        // Keep neighbors valid
        vec2 margin = uStepUV * (float(R) + 1.0);
        vec2 baseUV = clamp(vRawUV, vec2(margin.x, margin.y), vec2(1.0 - margin.x, 1.0 - margin.y));

        float dCenter = sampleDepthSigned(baseUV);

        // --- Edge strength (central difference with tiny step tied to uStepUV) ---
        vec2 gstep = uStepUV * uGradScale;
        float dpx = sampleDepthSigned(baseUV + vec2(+gstep.x, 0.0));
        float dmx = sampleDepthSigned(baseUV + vec2(-gstep.x, 0.0));
        float dpy = sampleDepthSigned(baseUV + vec2(0.0, +gstep.y));
        float dmy = sampleDepthSigned(baseUV + vec2(0.0, -gstep.y));
        float dx = dpx - dmx;
        float dy = dpy - dmy;
        float gradMag = sqrt(dx * dx + dy * dy);

        // Early-out: ignore flat/stretchy regions
        if (gradMag < uGradThresh) {
            FragColor = texture(textureSampler, vColorUV);
            return;
        }

        // --- Oriented near/far scan along the gradient direction ---
        vec2 gdir = vec2(dx, dy);
        float gnorm = length(gdir);
        if (gnorm > 1e-6) {
            gdir /= gnorm;
        } else {
            gdir = vec2(0.0, 0.0);
        }

        int Mr = uMarchR;
        if (Mr > R) {
            Mr = R;
        }

        float dNearRidge = dCenter;
        float dFarFloor  = dCenter;

        if (gnorm > 1e-6) {
            // March forward (+grad) to find a robust near-side ridge (max depth)
            for (int s = 1; s <= MAX_RADIUS; ++s) {
                if (s > Mr) break;
                vec2 uvF = baseUV + gdir * (uStepUV * float(s));
                float dF = sampleDepthSigned(uvF);
                dNearRidge = max(dNearRidge, dF);
            }
            // March backward (-grad) to find far-side floor (min depth)
            for (int s = 1; s <= MAX_RADIUS; ++s) {
                if (s > Mr) break;
                vec2 uvB = baseUV - gdir * (uStepUV * float(s));
                float dB = sampleDepthSigned(uvB);
                dFarFloor = min(dFarFloor, dB);
            }
        }

        // Near/far occlusion gap measured along the gradient direction
        float occGap = dNearRidge - dFarFloor;

        // One-sided discard: only if center is clearly behind the foreground ridge, and the near-far gap is large
        bool isBackgroundSide = (dCenter < (dNearRidge - uNearEps));
        bool strongOcclusion  = (occGap > uDiscardRange);

        if (isBackgroundSide && strongOcclusion) {
            discard;
        }

        FragColor = texture(textureSampler, vColorUV);
    }
    """

    frag_src = frag_template.replace("$MAX_RADIUS", str(max_radius))

    return vert_src, frag_src


# ============================= Renderer =============================

class QuiltRenderer:
    def __init__(self, max_radius: int, mesh_subdiv: int):
        vert_src, frag_src = make_shaders(max_radius)

        self.prog = link_program(vert_src, frag_src)

        self.loc_model = GL.glGetUniformLocation(self.prog, "model")
        self.loc_view = GL.glGetUniformLocation(self.prog, "view")
        self.loc_proj = GL.glGetUniformLocation(self.prog, "projection")
        self.loc_tex = GL.glGetUniformLocation(self.prog, "textureSampler")

        self.loc_R = GL.glGetUniformLocation(self.prog, "uRadius")
        self.loc_step = GL.glGetUniformLocation(self.prog, "uStepUV")
        self.loc_rng = GL.glGetUniformLocation(self.prog, "uDiscardRange")
        self.loc_ne_eps = GL.glGetUniformLocation(self.prog, "uNearEps")
        self.loc_ne_frac = GL.glGetUniformLocation(self.prog, "uMinNearFrac")

        # New oriented-edge uniforms
        self.loc_grad_scale = GL.glGetUniformLocation(self.prog, "uGradScale")
        self.loc_grad_thresh = GL.glGetUniformLocation(self.prog, "uGradThresh")
        self.loc_march_r = GL.glGetUniformLocation(self.prog, "uMarchR")

        GL.glUseProgram(self.prog)
        GL.glUniform1i(self.loc_tex, 0)

        # Sensible defaults (tunable from caller by reusing set_discard_params)
        if self.loc_grad_scale != -1:
            GL.glUniform1f(self.loc_grad_scale, 0.5)   # half of uStepUV for gradient sampling
        if self.loc_grad_thresh != -1:
            GL.glUniform1f(self.loc_grad_thresh, 0.02) # depth units ([-0.5..0.5])
        if self.loc_march_r != -1:
            GL.glUniform1i(self.loc_march_r, max_radius)

        self.vao, self.ebo, self.idx_count = make_plane(mesh_subdiv, mesh_subdiv)

    def set_discard_params(
        self,
        radius: int,
        step_uv_x: float,
        step_uv_y: float,
        discard_range: float,
        near_eps: float,
        min_near_frac: float,
        max_radius: int
    ) -> None:
        GL.glUseProgram(self.prog)

        R = int(min(max(radius, 1), max_radius))
        if self.loc_R != -1:
            GL.glUniform1i(self.loc_R, R)

        # Keep prior semantics: uStepUV is "half-size / R"; callers pass half-size in normalized UV.
        step_uv_x_norm = float(step_uv_x) / max(R, 1)
        step_uv_y_norm = float(step_uv_y) / max(R, 1)
        if self.loc_step != -1:
            GL.glUniform2f(self.loc_step, step_uv_x_norm, step_uv_y_norm)

        if self.loc_rng != -1:
            GL.glUniform1f(self.loc_rng, float(discard_range))

        if self.loc_ne_eps != -1:
            GL.glUniform1f(self.loc_ne_eps, float(near_eps))

        if self.loc_ne_frac != -1:
            GL.glUniform1f(self.loc_ne_frac, float(min_near_frac))

        # Keep oriented scan radius synchronized with neighborhood radius
        if self.loc_march_r != -1:
            GL.glUniform1i(self.loc_march_r, R)

        # Optional: adapt gradient threshold a bit to step size so it’s less sensitive to knob values
        if self.loc_grad_thresh != -1:
            # Target: a modest fraction of the occlusion gap per small step
            approx = max(0.01, 0.5 * float(discard_range))
            GL.glUniform1f(self.loc_grad_thresh, approx)

    def _common(self, W: int, H: int, cam_params: Dict[str, float]) -> Tuple[LKGCamera, float]:
        aspect = float(W) / float(H)
        cam = LKGCamera(
            size=cam_params["camera_size"],
            fov=cam_params["fov_deg"],
            viewcone=cam_params["viewcone_deg"],
            aspect=aspect,
            near=cam_params["near"],
            far=cam_params["far"]
        )
        return cam, aspect

    def render_quilt_to_fbo(
        self,
        combined_tex: int,
        W: int,
        H: int,
        cols: int,
        rows: int,
        cam_params: Dict[str, float],
        invert_quilt: bool,
        mesh_scale: float,
        yaw_deg: float,
        pitch_deg: float,
        roll_deg: float
    ) -> Tuple[int, int, int, int, int]:
        tile_w = W
        tile_h = H
        qw = cols * tile_w
        qh = rows * tile_h

        fbo, quilt_tex, quilt_rb = init_quilt_fbo(qw, qh)

        cam, aspect = self._common(W, H, cam_params)

        yaw = math.radians(yaw_deg)
        pitch = math.radians(pitch_deg)
        roll = math.radians(roll_deg)

        s = float(mesh_scale)
        model = _rot_z(roll) @ _rot_x(pitch) @ _rot_y(yaw) @ _scale(aspect * s, 1.0 * s, 1.0 * s)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        GL.glViewport(0, 0, qw, qh)
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        total = cols * rows

        GL.glUseProgram(self.prog)
        GL.glUniformMatrix4fv(self.loc_model, 1, GL.GL_TRUE, model)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, combined_tex)
        GL.glBindVertexArray(self.vao)

        for y in range(rows):
            for x in range(cols):
                idx = y * cols + x
                nrm = (idx / (total - 1)) if total > 1 else 0.5

                GL.glViewport(x * tile_w, y * tile_h, tile_w, tile_h)
                GL.glClear(GL.GL_DEPTH_BUFFER_BIT)

                V, P = cam.view_proj(nrm, bool(invert_quilt), cam_params["depthiness"], cam_params["focus"])

                GL.glUniformMatrix4fv(self.loc_view, 1, GL.GL_TRUE, V)
                GL.glUniformMatrix4fv(self.loc_proj, 1, GL.GL_TRUE, P)

                GL.glDrawElements(GL.GL_TRIANGLES, self.idx_count, GL.GL_UNSIGNED_INT, ctypes.c_void_p(0))

        GL.glBindVertexArray(0)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glFinish()

        return fbo, quilt_tex, quilt_rb, qw, qh

    def render_selected_views_to_fbo(
        self,
        combined_tex: int,
        W: int,
        H: int,
        view_norms: List[float],
        cam_params: Dict[str, float],
        invert_quilt: bool,
        mesh_scale: float,
        yaw_deg: float,
        pitch_deg: float,
        roll_deg: float
    ) -> Tuple[int, int, int, int, int]:
        clamped = [max(0.0, min(1.0, float(v))) for v in view_norms]
        out_w = max(1, len(clamped) * W)
        out_h = H

        fbo, out_tex, out_rb = init_quilt_fbo(out_w, out_h)

        cam, aspect = self._common(W, H, cam_params)

        yaw = math.radians(yaw_deg)
        pitch = math.radians(pitch_deg)
        roll = math.radians(roll_deg)

        s = float(mesh_scale)
        model = _rot_z(roll) @ _rot_x(pitch) @ _rot_y(yaw) @ _scale(aspect * s, 1.0 * s, 1.0 * s)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        GL.glViewport(0, 0, out_w, out_h)
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        GL.glUseProgram(self.prog)
        GL.glUniformMatrix4fv(self.loc_model, 1, GL.GL_TRUE, model)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, combined_tex)
        GL.glBindVertexArray(self.vao)

        for i, nrm in enumerate(clamped):
            GL.glViewport(i * W, 0, W, H)
            GL.glClear(GL.GL_DEPTH_BUFFER_BIT)

            V, P = cam.view_proj(nrm, bool(invert_quilt), cam_params["depthiness"], cam_params["focus"])

            GL.glUniformMatrix4fv(self.loc_view, 1, GL.GL_TRUE, V)
            GL.glUniformMatrix4fv(self.loc_proj, 1, GL.GL_TRUE, P)

            GL.glDrawElements(GL.GL_TRIANGLES, self.idx_count, GL.GL_UNSIGNED_INT, ctypes.c_void_p(0))

        GL.glBindVertexArray(0)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glFinish()

        return fbo, out_tex, out_rb, out_w, out_h

    def cleanup(self) -> None:
        GL.glDeleteBuffers(1, [self.ebo])
        GL.glDeleteVertexArrays(1, [self.vao])
        GL.glDeleteProgram(self.prog)
