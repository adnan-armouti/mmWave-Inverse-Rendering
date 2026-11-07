"""Evaluation of trained scenes: training RA, radar transfer, 3D occupancy and 3D reconstruction."""

# Fix EGL initialization for Open3D offscreen rendering.
# Mesa's swrast_dri.so needs GLIBCXX_3.4.30 from the system libstdc++,
# but conda's version may lack it. Preloading via ctypes fixes dlopen.
# IMPORTANT: Skip when Mitsuba/DrJit is loaded — RTLD_GLOBAL overrides
# their bundled libstdc++ symbols and causes a segfault during CUDA ops.
import os as _os
import sys as _sys
if "mitsuba" not in _sys.modules:
    import ctypes as _ctypes
    _sys_libstdcxx = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
    if _os.path.isfile(_sys_libstdcxx):
        try:
            _ctypes.CDLL(_sys_libstdcxx, mode=_ctypes.RTLD_GLOBAL)
        except OSError:
            pass
if "LIBGL_DRIVERS_PATH" not in _os.environ:
    _dri_path = "/usr/lib/x86_64-linux-gnu/dri"
    if _os.path.isdir(_dri_path):
        _os.environ["LIBGL_DRIVERS_PATH"] = _dri_path
