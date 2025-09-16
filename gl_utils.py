# gl_utils.py
import os, sys, math, ctypes, platform
from typing import Tuple, Optional
import numpy as np
import glfw
from OpenGL import GL

def compile_shader(src: str, shader_type) -> int:
    sid = GL.glCreateShader(shader_type)
    GL.glShaderSource(sid, src)
    GL.glCompileShader(sid)
    if not GL.glGetShaderiv(sid, GL.GL_COMPILE_STATUS):
        log = GL.glGetShaderInfoLog(sid).decode()
        GL.glDeleteShader(sid)
        kind = "vertex" if shader_type == GL.GL_VERTEX_SHADER else "fragment"
        numbered = "\n".join(f"{i+1:4d}: {l}" for i, l in enumerate(src.splitlines()))
        raise RuntimeError(f"{kind} shader compile failed:\n{log}\n---\n{numbered}")
    return sid

def link_program(vs_src: str, fs_src: str) -> int:
    vs = compile_shader(vs_src, GL.GL_VERTEX_SHADER)
    fs = compile_shader(fs_src, GL.GL_FRAGMENT_SHADER)
    prog = GL.glCreateProgram()
    GL.glAttachShader(prog, vs)
    GL.glAttachShader(prog, fs)
    GL.glLinkProgram(prog)
    if not GL.glGetProgramiv(prog, GL.GL_LINK_STATUS):
        log = GL.glGetProgramInfoLog(prog).decode()
        GL.glDeleteProgram(prog); GL.glDeleteShader(vs); GL.glDeleteShader(fs)
        raise RuntimeError(f"Program link failed:\n{log}")
    GL.glDeleteShader(vs); GL.glDeleteShader(fs)
    return prog

def create_gl_context(title: str, visible: bool, width: int, height: int, attempts=((4,6),)) -> glfw._GLFWwindow:
    if not glfw.init():
        raise RuntimeError("GLFW init failed")
    glfw.window_hint(glfw.VISIBLE, glfw.TRUE if visible else glfw.FALSE)
    glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
    glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)
    is_macos = (platform.system() == "Darwin")
    window = None
    last_err = None
    for (maj, minr) in attempts:
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
            window = glfw.create_window(width, height, f"{title} GL {maj}.{minr}", None, None)
            if not window:
                last_err = glfw.get_error(); continue
            glfw.make_context_current(window)
            ver = GL.glGetString(GL.GL_VERSION)
            print(f"[GL] Context: {ver.decode() if hasattr(ver,'decode') else ver} (requested {maj}.{minr})")
            break
        except Exception as e:
            last_err = e; continue
    if window is None:
        glfw.terminate()
        raise RuntimeError(f"Failed to create GLFW window. Last error: {last_err}")
    return window

def create_texture_rgba(img_rgba: np.ndarray, filter_linear: bool = True) -> int:
    h, w, c = img_rgba.shape; assert c == 4
    tex = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR if filter_linear else GL.GL_NEAREST)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR if filter_linear else GL.GL_NEAREST)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, w, h, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, img_rgba)
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    return tex

def update_texture_rgba(tex: int, img_rgba: np.ndarray, filter_linear: bool = True) -> None:
    h, w, c = img_rgba.shape; assert c == 4
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR if filter_linear else GL.GL_NEAREST)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR if filter_linear else GL.GL_NEAREST)
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, w, h, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, img_rgba)
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

def delete_texture(tex: Optional[int]) -> None:
    if tex:
        GL.glDeleteTextures(1, [tex])

def make_fullscreen_present() -> Tuple[int,int,int,int,int,int,int]:
    VS = r"""
    #version 330 core
    layout(location=0) in vec2 inPos;
    layout(location=1) in vec2 inUV;
    out vec2 vUV;
    void main(){ gl_Position = vec4(inPos,0.0,1.0); vUV = inUV; }
    """
    FS = r"""
    #version 330 core
    in vec2 vUV;
    out vec4 FragColor;
    uniform sampler2D uTex;
    uniform int uFlipY;
    void main(){ vec2 t=vUV; if(uFlipY==1) t.y=1.0-t.y; FragColor = texture(uTex,t); }
    """
    prog = link_program(VS, FS)
    loc_tex = GL.glGetUniformLocation(prog, "uTex")
    loc_flip = GL.glGetUniformLocation(prog, "uFlipY")
    verts = np.array([-1.0,-1.0,0.0,0.0, 1.0,-1.0,1.0,0.0, 1.0,1.0,1.0,1.0, -1.0,1.0,0.0,1.0], dtype=np.float32)
    idx = np.array([0,1,2, 0,2,3], dtype=np.uint32)
    vao = GL.glGenVertexArrays(1); vbo = GL.glGenBuffers(1); ebo = GL.glGenBuffers(1)
    GL.glBindVertexArray(vao)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo); GL.glBufferData(GL.GL_ARRAY_BUFFER, verts.nbytes, verts, GL.GL_STATIC_DRAW)
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ebo); GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL.GL_STATIC_DRAW)
    stride = 4*4
    GL.glEnableVertexAttribArray(0); GL.glVertexAttribPointer(0,2,GL.GL_FLOAT,GL.GL_FALSE,stride,ctypes.c_void_p(0))
    GL.glEnableVertexAttribArray(1); GL.glVertexAttribPointer(1,2,GL.GL_FLOAT,GL.GL_FALSE,stride,ctypes.c_void_p(8))
    GL.glBindVertexArray(0)
    return prog, loc_tex, loc_flip, vao, vbo, ebo, 6

def present_texture(prog: int, loc_tex: int, loc_flip: int, vao: int, index_count: int, tex: int, flip_y: bool) -> None:
    GL.glUseProgram(prog)
    GL.glUniform1i(loc_tex, 0)
    GL.glUniform1i(loc_flip, 1 if flip_y else 0)
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
    GL.glBindVertexArray(vao)
    GL.glDrawElements(GL.GL_TRIANGLES, index_count, GL.GL_UNSIGNED_INT, ctypes.c_void_p(0))
    GL.glBindVertexArray(0); GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

def init_quilt_fbo(qw: int, qh: int) -> Tuple[int,int,int]:
    tex = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, qw, qh, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
    rb = GL.glGenRenderbuffers(1); GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, rb)
    GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT24, qw, qh)
    fbo = GL.glGenFramebuffers(1); GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
    GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, tex, 0)
    GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, rb)
    if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError("FBO incomplete")
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
    return fbo, tex, rb

def read_fbo_rgba(fbo: int, w: int, h: int) -> np.ndarray:
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
    GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
    data = GL.glReadPixels(0, 0, w, h, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
    arr = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 4))
    return np.flipud(arr)
