"""
FlashDetect Python Wrapper — safe, Pythonic API on top of flashdetect.dll
"""
import ctypes
import os
import glob
from typing import List, Optional, Tuple

# Add tensorrt_libs + CUDA runtime DLL dirs to search path if installed
try:
    import tensorrt_libs
    os.add_dll_directory(tensorrt_libs.__path__[0])
except ImportError:
    pass
# Scan all nvidia/* packages for DLL directories
_nvidia_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'nvidia')
if os.path.isdir(_nvidia_base):
    for _root, _dirs, _files in os.walk(_nvidia_base):
        if any(f.endswith('.dll') for f in _files):
            os.add_dll_directory(_root)


class Detection:
    """Single detection result."""
    __slots__ = ("x1", "y1", "x2", "y2", "conf", "class_id")

    def __init__(self, x1: float, y1: float, x2: float, y2: float,
                 conf: float, class_id: int):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.conf = conf
        self.class_id = class_id

    def __repr__(self):
        return (f"Detection(x1={self.x1:.0f}, y1={self.y1:.0f}, "
                f"x2={self.x2:.0f}, y2={self.y2:.0f}, "
                f"cls={self.class_id}, conf={self.conf:.2f})")

    @property
    def xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


class FlashDetect:
    """
    Low-latency YOLO26 inference engine.
    """

    def __init__(
        self,
        engine_path: str,
        conf: float = 0.25,
        device_id: int = 0,
        target_classes: Optional[List[int]] = None,
        max_dets: int = 0,
        format: str = "BGR",
        resize_mode: int = 0,       # 0=no resize (default), 1=GPU resize
        src_w: int = 0,             # source width (resize_mode=1)
        src_h: int = 0,             # source height (resize_mode=1)
        sdk_dir: Optional[str] = None,
    ):
        if sdk_dir is None:
            # Auto-detect: look for flashdetect.dll next to this file
            sdk_dir = os.path.dirname(os.path.abspath(__file__))
            # Also try parent directory's runtime folder
            parent_runtime = os.path.join(os.path.dirname(sdk_dir), "runtime")
            if os.path.isdir(parent_runtime):
                os.add_dll_directory(parent_runtime)
            os.add_dll_directory(sdk_dir)

        # Find DLL
        dll_path = os.path.join(sdk_dir, "flashdetect.dll")
        if not os.path.exists(dll_path):
            raise FileNotFoundError(f"flashdetect.dll not found in {sdk_dir}")

        try:
            self._dll = ctypes.CDLL(dll_path)
        except OSError as e:
            raise RuntimeError(
                f"Cannot load flashdetect.dll — missing runtime DLLs.\n"
                f"  {e}\n"
                f"  Ensure CUDA 12.4 + TensorRT 11.0 are installed."
            ) from e
        self._setup_ctypes()

        # Resolve engine path (support glob)
        if not os.path.exists(engine_path):
            candidates = glob.glob(engine_path)
            if not candidates:
                candidates = glob.glob(os.path.join(os.path.dirname(sdk_dir), "*.engine"))
            if candidates:
                engine_path = candidates[0]
            else:
                raise FileNotFoundError(f"Engine not found: {engine_path}")

        # Prepare target classes
        n_labels = len(target_classes) if target_classes else 0
        c_labels = (ctypes.c_int * n_labels)(*target_classes) if target_classes else None

        # Convert format string to DLL value
        fmt_map = {"BGR": 0, "RGB": 1}
        fmt_val = fmt_map.get(format.upper(), 0)

        # Create context
        self._ctx = self._dll.fd_create(
            engine_path.encode("utf-8"),
            ctypes.c_float(conf),
            ctypes.c_int(device_id),
            c_labels,
            ctypes.c_int(n_labels),
            ctypes.c_int(max_dets),
            ctypes.c_int(fmt_val),
            ctypes.c_int(resize_mode),
            ctypes.c_int(src_w),
            ctypes.c_int(src_h),
        )
        if not self._ctx:
            # Try to get machine ID to guide the user
            try:
                mid = get_machine_id()
            except Exception:
                mid = "<unavailable>"
            raise RuntimeError(
                f"fd_create failed — likely missing or invalid license.key\n"
                f"  Your machine ID: {mid}\n"
                f"  Send this ID to the vendor to obtain a license.key file,\n"
                f"  then place it beside flashdetect.dll.\n"
                f"  Or run: python -c \"import flashdetect; print(flashdetect.get_machine_id())\""
            )

        # Get engine input size
        h, w = ctypes.c_int(), ctypes.c_int()
        self._dll.fd_get_size(self._ctx, ctypes.byref(h), ctypes.byref(w))
        self.input_height = h.value
        self.input_width = w.value
        self.resize_mode = resize_mode
        self._src_w = src_w if src_w > 0 else self.input_width
        self._src_h = src_h if src_h > 0 else self.input_height

    # ── ctypes setup ──────────────────────────────

    def _setup_ctypes(self):
        d = self._dll

        class _Ctx(ctypes.Structure):
            pass

        class _Det(ctypes.Structure):
            _fields_ = [
                ("x1", ctypes.c_float), ("y1", ctypes.c_float),
                ("x2", ctypes.c_float), ("y2", ctypes.c_float),
                ("conf", ctypes.c_float), ("class_id", ctypes.c_int),
            ]

        d.fd_create.argtypes = [
            ctypes.c_char_p, ctypes.c_float, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        d.fd_create.restype = ctypes.POINTER(_Ctx)

        d.fd_process.argtypes = [
            ctypes.POINTER(_Ctx), ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(_Det), ctypes.c_int, ctypes.POINTER(ctypes.c_int),
        ]
        d.fd_process.restype = ctypes.c_int

        d.fd_get_size.argtypes = [
            ctypes.POINTER(_Ctx), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        d.fd_get_size.restype = None

        d.fd_release.argtypes = [ctypes.POINTER(_Ctx)]
        d.fd_release.restype = None

        d.fd_set_conf.argtypes = [ctypes.POINTER(_Ctx), ctypes.c_float]
        d.fd_set_conf.restype = None

        d.fd_set_classes.argtypes = [ctypes.POINTER(_Ctx), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        d.fd_set_classes.restype = None

        self._Det = _Det

    # ── Public API ────────────────────────────────

    def detect(self, image_rgb, conf: float = None, classes: List[int] = None) -> List[Detection]:
        """
        Run inference on a single image.

        Args:
            image_rgb: numpy array (H, W, 3) uint8 RGB.
            conf:      Override confidence threshold for this frame (None = keep current).
            classes:   Override target classes for this frame (None = keep current).

        Returns:
            List of Detection objects.
        """
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
            c_cls = (ctypes.c_int * n)(*classes)
            self._dll.fd_set_classes(self._ctx, c_cls, ctypes.c_int(n))

        dets = (self._Det * 300)()
        count = ctypes.c_int()
        ptr = image_rgb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        ret = self._dll.fd_process(self._ctx, ptr, dets, 300, ctypes.byref(count))
        if ret != 0:
            raise RuntimeError(f"fd_process failed with code {ret}")

        results = []
        sx = self._src_w / self.input_width  if self.resize_mode else 1.0
        sy = self._src_h / self.input_height if self.resize_mode else 1.0
        for i in range(count.value):
            d = dets[i]
            b = Detection(d.x1*sx, d.y1*sy, d.x2*sx, d.y2*sy, d.conf, d.class_id)
            results.append(b)
        return results

    def __call__(self, image) -> List[Detection]:
        """Shortcut: engine(image) → detections"""
        return self.detect(image)

    def close(self):
        """Release GPU resources."""
        if self._ctx:
            self._dll.fd_release(self._ctx)
            self._ctx = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        """Auto-release on garbage collection."""
        try:
            self.close()
        except Exception:
            pass

    @property
    def input_size(self) -> Tuple[int, int]:
        return (self.input_height, self.input_width)


# ── Module-level utilities ──────────────────────

def get_machine_id(sdk_dir: str = None) -> str:
    """
    Get this machine's unique hardware ID for license generation.

    Usage (first-time setup):
        >>> import flashdetect
        >>> print(flashdetect.get_machine_id())
        A1B2C3D4E5F67890

    Send this ID to the vendor to receive a license.key file,
    then place it beside flashdetect.dll (in the same directory).

    Args:
        sdk_dir: Path to the directory containing flashdetect.dll.
                 Defaults to the package install directory.
    Returns:
        16-character hex machine ID string.
    """
    if sdk_dir is None:
        sdk_dir = os.path.dirname(os.path.abspath(__file__))
        os.add_dll_directory(sdk_dir)

    dll_path = os.path.join(sdk_dir, "flashdetect.dll")
    if not os.path.exists(dll_path):
        raise FileNotFoundError(f"flashdetect.dll not found in {sdk_dir}")

    try:
        dll = ctypes.CDLL(dll_path)
    except OSError as e:
        raise RuntimeError(
            f"Cannot load flashdetect.dll — missing runtime DLLs.\n"
            f"  {e}\n"
            f"  Ensure CUDA 12.4 + TensorRT 11.0 are installed."
        ) from e
    dll.fd_get_machine_id.argtypes = [ctypes.c_char_p, ctypes.c_int]
    dll.fd_get_machine_id.restype = ctypes.c_int

    buf = ctypes.create_string_buffer(32)
    if dll.fd_get_machine_id(buf, 32) != 0:
        raise RuntimeError("Failed to get machine ID")
    return buf.value.decode("utf-8")


def install_license(source_path: str, sdk_dir: str = None) -> str:
    """
    Copy a license.key file into the flashdetect package directory.

    Usage:
        >>> import flashdetect
        >>> flashdetect.install_license("C:/Users/me/Downloads/license.key")

    This copies the file to the same directory as flashdetect.dll, where
    the DLL looks for it at runtime.

    Args:
        source_path: Path to your license.key file.
        sdk_dir:     Target directory (default: flashdetect package dir).
    Returns:
        The destination path where license.key was installed.
    """
    import shutil

    if not os.path.exists(source_path):
        raise FileNotFoundError(f"license.key not found at: {source_path}")

    if sdk_dir is None:
        sdk_dir = os.path.dirname(os.path.abspath(__file__))

    dest_path = os.path.join(sdk_dir, "license.key")

    if os.path.exists(dest_path):
        print(f"[flashdetect] Replacing existing: {dest_path}")

    shutil.copy2(source_path, dest_path)
    print(f"[flashdetect] license.key installed → {dest_path}")

    mid = get_machine_id(sdk_dir)
    print(f"[flashdetect] Machine ID: {mid}")

    return dest_path
