"""FlashDetect Python Wrapper — unified Win/Linux with auto-download"""
import ctypes
import os
import sys
import glob
import zipfile
import tempfile
import shutil
from typing import List, Optional, Tuple

__version__ = "1.0.3"

_GITHUB_REPO = "https://github.com/toever/flashdetect"
_VERSION = "1.0.3"
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))

def _get_platform_tag():
    if sys.platform == "win32":
        return "win_amd64"
    else:
        return "manylinux_2_28_x86_64"

def _get_dll_name():
    return "flashdetect.dll" if sys.platform == "win32" else "libflashdetect.so"

def _ensure_native():
    dll = _get_dll_name()
    dll_path = os.path.join(_PKG_DIR, dll)
    if os.path.exists(dll_path):
        return
    print("[flashdetect] Native library not found, downloading from GitHub...")
    plat = _get_platform_tag()
    wheel_name = "flashdetect_trt111_cu124-{}-py3-none-{}.whl".format(_VERSION, plat)
    url = "{}/releases/download/v{}/{}".format(_GITHUB_REPO, _VERSION, wheel_name)
    tmpdir = tempfile.mkdtemp(prefix="flashdetect_dl_")
    try:
        wheel_path = os.path.join(tmpdir, wheel_name)
        _download(url, wheel_path)
        _extract_native(wheel_path, _PKG_DIR)
        print("[flashdetect] Download complete.")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def _download(url, dst):
    try:
        from urllib.request import urlopen, Request
        req = Request(url, headers={"User-Agent": "flashdetect-pypi"})
        with urlopen(req) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            with open(dst, "wb") as f:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded * 100 // total
                        mb = downloaded / (1 << 20)
                        total_mb = total / (1 << 20)
                        print("\r  Downloading... {}% ({:.1f}/{:.1f} MB)".format(pct, mb, total_mb), end="")
                if total:
                    print()
    except Exception as e:
        raise RuntimeError(
            "Failed to download native library from {}\nError: {}\n"
            "Please manually download from:\n  {}/releases/tag/v{}\n"
            "Then run: pip install <downloaded.whl>".format(
                url, e, _GITHUB_REPO, _VERSION)) from e

def _extract_native(wheel_path, dst_dir):
    with zipfile.ZipFile(wheel_path, "r") as zf:
        for member in zf.namelist():
            if not member.startswith("flashdetect/"):
                continue
            name = os.path.basename(member)
            if member.endswith("/"):
                continue
            parts = member.split("/")
            if len(parts) > 2 and parts[1] == "libs":
                target = os.path.join(dst_dir, "libs", name)
            elif len(parts) == 2 and (name.endswith(".dll") or name.endswith(".so") or name == "__init__.py"):
                target = os.path.join(dst_dir, name)
            else:
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())

_ensure_native()

_libs_dir = os.path.join(_PKG_DIR, "libs")

if sys.platform == "win32":
    os.add_dll_directory(_PKG_DIR)
    if os.path.isdir(_libs_dir):
        os.add_dll_directory(_libs_dir)

elif sys.platform in ("linux", "linux2"):
    pass

class Detection:
    __slots__ = ("x1", "y1", "x2", "y2", "conf", "class_id")
    def __init__(self, x1, y1, x2, y2, conf, class_id):
        self.x1 = x1; self.y1 = y1; self.x2 = x2; self.y2 = y2
        self.conf = conf; self.class_id = class_id
    def __repr__(self):
        return ("Detection(x1={:.0f}, y1={:.0f}, x2={:.0f}, y2={:.0f}, "
                "cls={}, conf={:.2f})".format(
                    self.x1, self.y1, self.x2, self.y2,
                    self.class_id, self.conf))
    @property
    def xyxy(self):
        return (self.x1, self.y1, self.x2, self.y2)

class FlashDetect:
    def __init__(self, engine_path, conf=0.25, device_id=0,
                 target_classes=None, max_dets=0, format="BGR",
                 resize_mode=1, sdk_dir=None):
        if sdk_dir is None:
            sdk_dir = _PKG_DIR
        dll_name = _get_dll_name()
        dll_path = os.path.join(sdk_dir, dll_name)
        if not os.path.exists(dll_path):
            raise FileNotFoundError(
                "{} not found in {}\n"
                "Re-run: pip uninstall flashdetect && pip install flashdetect".format(
                    dll_name, sdk_dir))
        try:
            self._dll = ctypes.CDLL(dll_path)
        except OSError as e:
            raise RuntimeError("Cannot load {}: {}".format(dll_name, e)) from e
        self._setup_ctypes()
        if not os.path.exists(engine_path):
            candidates = (glob.glob(engine_path) or
                          glob.glob(os.path.join(os.path.dirname(sdk_dir), "*.engine")))
            if candidates:
                engine_path = candidates[0]
            else:
                raise FileNotFoundError("Engine not found: {}".format(engine_path))
        n_labels = len(target_classes) if target_classes else 0
        c_labels = (ctypes.c_int * n_labels)(*target_classes) if target_classes else None
        fmt_val = {"BGR": 0, "RGB": 1}.get(format.upper(), 0)
        self._ctx = self._dll.fd_create(
            engine_path.encode(), ctypes.c_float(conf),
            ctypes.c_int(device_id), c_labels, ctypes.c_int(n_labels),
            ctypes.c_int(max_dets), ctypes.c_int(fmt_val),
            ctypes.c_int(resize_mode))
        if not self._ctx:
            try:
                mid = get_machine_id()
            except Exception:
                mid = "<unavailable>"
            raise RuntimeError(
                "fd_create failed. Possible causes:\n"
                "  1. Invalid or missing license.key (Machine ID: {})\n"
                "  2. Engine file corrupted or built with incompatible TensorRT version\n"
                "  3. GPU driver / CUDA version mismatch\n"
                "Check console output above for [ERR] messages.".format(mid))
        h, w = ctypes.c_int(), ctypes.c_int()
        self._dll.fd_get_size(self._ctx, ctypes.byref(h), ctypes.byref(w))
        self.input_height = h.value
        self.input_width = w.value
        self.resize_mode = resize_mode
        self._src_w = self.input_width
        self._src_h = self.input_height
        self._max_dets = max_dets if max_dets > 0 else 300

    def _setup_ctypes(self):
        d = self._dll
        class _Ctx(ctypes.Structure):
            pass
        class _Det(ctypes.Structure):
            _fields_ = [
                ("x1", ctypes.c_float), ("y1", ctypes.c_float),
                ("x2", ctypes.c_float), ("y2", ctypes.c_float),
                ("conf", ctypes.c_float), ("class_id", ctypes.c_int)]
        d.fd_create.argtypes = [
            ctypes.c_char_p, ctypes.c_float, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int]
        d.fd_create.restype = ctypes.POINTER(_Ctx)
        d.fd_process.argtypes = [
            ctypes.POINTER(_Ctx), ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(_Det), ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        d.fd_process.restype = ctypes.c_int
        d.fd_get_size.argtypes = [
            ctypes.POINTER(_Ctx), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int)]
        d.fd_get_size.restype = None
        d.fd_release.argtypes = [ctypes.POINTER(_Ctx)]
        d.fd_release.restype = None
        d.fd_set_conf.argtypes = [ctypes.POINTER(_Ctx), ctypes.c_float]
        d.fd_set_conf.restype = None
        d.fd_set_classes.argtypes = [
            ctypes.POINTER(_Ctx), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        d.fd_set_classes.restype = None
        d.fd_set_src_size.argtypes = [
            ctypes.POINTER(_Ctx), ctypes.c_int, ctypes.c_int]
        d.fd_set_src_size.restype = None
        self._Det = _Det

    def detect(self, image_rgb, conf=None, classes=None, max_dets=None):
        import numpy as np
        if not isinstance(image_rgb, np.ndarray):
            raise TypeError("image_rgb must be a numpy array")
        if image_rgb.dtype != np.uint8:
            image_rgb = image_rgb.astype(np.uint8)
        if not image_rgb.flags["C_CONTIGUOUS"]:
            image_rgb = np.ascontiguousarray(image_rgb)
        if conf is not None:
            self._dll.fd_set_conf(self._ctx, ctypes.c_float(conf))
        if classes is not None:
            n = len(classes)
            self._dll.fd_set_classes(
                self._ctx, (ctypes.c_int * n)(*classes), ctypes.c_int(n))
        if self.resize_mode:
            h, w = image_rgb.shape[:2]
            if w != self._src_w or h != self._src_h:
                self._dll.fd_set_src_size(
                    self._ctx, ctypes.c_int(w), ctypes.c_int(h))
                self._src_w, self._src_h = w, h
        n_dets = self._max_dets if max_dets is None else min(max_dets, self._max_dets)
        dets = (self._Det * n_dets)()
        count = ctypes.c_int()
        ptr = image_rgb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        if self._dll.fd_process(self._ctx, ptr, dets, n_dets,
                                ctypes.byref(count)) != 0:
            raise RuntimeError("fd_process failed")
        return [Detection(d.x1, d.y1, d.x2, d.y2, d.conf, d.class_id)
                for d in dets[:count.value]]

    def __call__(self, image):
        return self.detect(image)

    def close(self):
        if self._ctx:
            self._dll.fd_release(self._ctx)
            self._ctx = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @property
    def input_size(self):
        return (self.input_height, self.input_width)

def get_machine_id(sdk_dir=None):
    if sdk_dir is None:
        sdk_dir = _PKG_DIR
    dll_name = _get_dll_name()
    dll_path = os.path.join(sdk_dir, dll_name)
    if not os.path.exists(dll_path):
        raise FileNotFoundError("{} not found in {}".format(dll_name, sdk_dir))
    try:
        dll = ctypes.CDLL(dll_path)
    except OSError as e:
        raise RuntimeError("Cannot load {}: {}".format(dll_name, e)) from e
    dll.fd_get_machine_id.argtypes = [ctypes.c_char_p, ctypes.c_int]
    dll.fd_get_machine_id.restype = ctypes.c_int
    buf = ctypes.create_string_buffer(32)
    if dll.fd_get_machine_id(buf, 32) != 0:
        raise RuntimeError("Failed to get machine ID")
    return buf.value.decode()


def install_license(path):
    """Copy license.key to the flashdetect package directory."""
    import shutil
    dst = os.path.join(_PKG_DIR, "license.key")
    shutil.copy2(path, dst)
    print(f"[flashdetect] License installed: {dst}")
