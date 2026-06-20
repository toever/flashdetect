"""FlashDetect Python Wrapper (Linux)"""
import ctypes
import os
import glob
from typing import List, Optional, Tuple

_my_dir = os.path.dirname(os.path.abspath(__file__))
_libs = os.path.join(_my_dir, "libs")
if os.path.isdir(_libs):
    for _f in sorted(os.listdir(_libs)):
        if _f.endswith('.so'):
            try:
                ctypes.CDLL(os.path.realpath(os.path.join(_libs, _f)), ctypes.RTLD_GLOBAL)
            except OSError:
                pass

class Detection:
    __slots__ = ("x1", "y1", "x2", "y2", "conf", "class_id")
    def __init__(self, x1, y1, x2, y2, conf, class_id):
        self.x1=x1; self.y1=y1; self.x2=x2; self.y2=y2
        self.conf=conf; self.class_id=class_id
    def __repr__(self):
        return (f"Detection(x1={self.x1:.0f}, y1={self.y1:.0f}, "
                f"x2={self.x2:.0f}, y2={self.y2:.0f}, "
                f"cls={self.class_id}, conf={self.conf:.2f})")
    @property
    def xyxy(self): return (self.x1, self.y1, self.x2, self.y2)

class FlashDetect:
    def __init__(self, engine_path: str, conf=0.25, device_id=0,
                 target_classes=None, max_dets=0, format="BGR",
                 resize_mode=1, src_w=0, src_h=0, sdk_dir=None):
        if sdk_dir is None:
            sdk_dir = os.path.dirname(os.path.abspath(__file__))
        dll_path = os.path.join(sdk_dir, "libflashdetect.so")
        if not os.path.exists(dll_path):
            raise FileNotFoundError(f"libflashdetect.so not found in {sdk_dir}")
        try: self._dll = ctypes.CDLL(dll_path)
        except OSError as e: raise RuntimeError(f"Cannot load libflashdetect.so: {e}") from e
        self._setup_ctypes()
        if not os.path.exists(engine_path):
            candidates = glob.glob(engine_path) or glob.glob(os.path.join(os.path.dirname(sdk_dir), "*.engine"))
            if candidates: engine_path = candidates[0]
            else: raise FileNotFoundError(f"Engine not found: {engine_path}")
        n_labels = len(target_classes) if target_classes else 0
        c_labels = (ctypes.c_int * n_labels)(*target_classes) if target_classes else None
        fmt_val = {"BGR":0,"RGB":1}.get(format.upper(),0)
        self._ctx = self._dll.fd_create(engine_path.encode(), ctypes.c_float(conf),
            ctypes.c_int(device_id), c_labels, ctypes.c_int(n_labels),
            ctypes.c_int(max_dets), ctypes.c_int(fmt_val), ctypes.c_int(resize_mode),
            ctypes.c_int(src_w), ctypes.c_int(src_h))
        if not self._ctx:
            try: mid = get_machine_id()
            except Exception: mid = "<unavailable>"
            raise RuntimeError(f"fd_create failed - check license.key. Machine ID: {mid}")
        h,w = ctypes.c_int(), ctypes.c_int()
        self._dll.fd_get_size(self._ctx, ctypes.byref(h), ctypes.byref(w))
        self.input_height=h.value; self.input_width=w.value
        self.resize_mode=resize_mode
        self._src_w=src_w if src_w>0 else self.input_width
        self._src_h=src_h if src_h>0 else self.input_height
        self._max_dets=max_dets if max_dets>0 else 300

    def _setup_ctypes(self):
        d=self._dll
        class _Ctx(ctypes.Structure): pass
        class _Det(ctypes.Structure):
            _fields_=[("x1",ctypes.c_float),("y1",ctypes.c_float),("x2",ctypes.c_float),
                      ("y2",ctypes.c_float),("conf",ctypes.c_float),("class_id",ctypes.c_int)]
        d.fd_create.argtypes=[ctypes.c_char_p,ctypes.c_float,ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),ctypes.c_int,ctypes.c_int,ctypes.c_int,
            ctypes.c_int,ctypes.c_int,ctypes.c_int]
        d.fd_create.restype=ctypes.POINTER(_Ctx)
        d.fd_process.argtypes=[ctypes.POINTER(_Ctx),ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(_Det),ctypes.c_int,ctypes.POINTER(ctypes.c_int)]
        d.fd_process.restype=ctypes.c_int
        d.fd_get_size.argtypes=[ctypes.POINTER(_Ctx),ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int)]
        d.fd_get_size.restype=None
        d.fd_release.argtypes=[ctypes.POINTER(_Ctx)]; d.fd_release.restype=None
        d.fd_set_conf.argtypes=[ctypes.POINTER(_Ctx),ctypes.c_float]; d.fd_set_conf.restype=None
        d.fd_set_classes.argtypes=[ctypes.POINTER(_Ctx),ctypes.POINTER(ctypes.c_int),ctypes.c_int]; d.fd_set_classes.restype=None
        d.fd_set_src_size.argtypes=[ctypes.POINTER(_Ctx),ctypes.c_int,ctypes.c_int]; d.fd_set_src_size.restype=None
        self._Det=_Det

    def detect(self, image_rgb, conf=None, classes=None):
        import numpy as np
        if not isinstance(image_rgb, np.ndarray): raise TypeError("image_rgb must be a numpy array")
        if image_rgb.dtype!=np.uint8: image_rgb=image_rgb.astype(np.uint8)
        if not image_rgb.flags["C_CONTIGUOUS"]: image_rgb=np.ascontiguousarray(image_rgb)
        if conf is not None: self._dll.fd_set_conf(self._ctx, ctypes.c_float(conf))
        if classes is not None:
            n=len(classes); self._dll.fd_set_classes(self._ctx, (ctypes.c_int*n)(*classes), ctypes.c_int(n))
        if self.resize_mode:
            h,w=image_rgb.shape[:2]; self._dll.fd_set_src_size(self._ctx, ctypes.c_int(w), ctypes.c_int(h))
        dets=(self._Det*self._max_dets)(); count=ctypes.c_int()
        ptr=image_rgb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        if self._dll.fd_process(self._ctx, ptr, dets, self._max_dets, ctypes.byref(count))!=0:
            raise RuntimeError("fd_process failed")
        return [Detection(d.x1,d.y1,d.x2,d.y2,d.conf,d.class_id) for d in dets[:count.value]]

    def __call__(self, image): return self.detect(image)
    def close(self):
        if self._ctx: self._dll.fd_release(self._ctx); self._ctx=None
    def __enter__(self): return self
    def __exit__(self,*a): self.close()
    def __del__(self):
        try: self.close()
        except: pass
    @property
    def input_size(self): return (self.input_height, self.input_width)

def get_machine_id(sdk_dir=None):
    if sdk_dir is None: sdk_dir=os.path.dirname(os.path.abspath(__file__))
    dll_path=os.path.join(sdk_dir,"libflashdetect.so")
    if not os.path.exists(dll_path): raise FileNotFoundError(f"libflashdetect.so not found in {sdk_dir}")
    try: dll=ctypes.CDLL(dll_path)
    except OSError as e: raise RuntimeError(f"Cannot load libflashdetect.so: {e}") from e
    dll.fd_get_machine_id.argtypes=[ctypes.c_char_p,ctypes.c_int]
    dll.fd_get_machine_id.restype=ctypes.c_int
    buf=ctypes.create_string_buffer(32)
    if dll.fd_get_machine_id(buf,32)!=0: raise RuntimeError("Failed to get machine ID")
    return buf.value.decode()

def install_license(source_path, sdk_dir=None):
    import shutil
    if not os.path.exists(source_path): raise FileNotFoundError(f"license.key not found: {source_path}")
    if sdk_dir is None: sdk_dir=os.path.dirname(os.path.abspath(__file__))
    dest=os.path.join(sdk_dir,"license.key")
    if os.path.exists(dest): print(f"[flashdetect] Replacing: {dest}")
    shutil.copy2(source_path,dest)
    print(f"[flashdetect] license.key -> {dest}")
    print(f"[flashdetect] Machine ID: {get_machine_id(sdk_dir)}")
    return dest
