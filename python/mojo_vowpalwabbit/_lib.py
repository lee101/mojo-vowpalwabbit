"""ctypes bindings for the Mojo kernels."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "vowpalwabbit.mojo")
LIB = os.environ.get("MOJO_VOWPALWABBIT_LIB") or os.path.join(
    ROOT, "dist", "libmojo-vowpalwabbit.so"
)

I = ctypes.c_int64
F = ctypes.c_float
P = ctypes.c_char_p

_SIGNATURES = {
    "mvw_uniform_hash": ([I, I, I], I),
    "mvw_hash_string": ([I, I, I, I], I),
    "mvw_hash_many": ([I, I, I, I, I, I, I, I, I], None),
    "mvw_parse_line": ([P] + [I] * 10, I),
    "mvw_learn_text": ([P] + [I] * 15 + [F, F, F, F] + [I] * 8, I),
    "mvw_predict_many": ([I, I, I, I, I, I, I, I, F, F, I], None),
    "mvw_learn_many": (
        [I] * 12 + [I, I, F, F, F, F] + [I] * 6,
        None,
    ),
}


class BuildError(RuntimeError):
    pass


def _mojo_command() -> list[str]:
    override = os.environ.get("MOJO_VOWPALWABBIT_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi")
    if pixi:
        return [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "mojo"]
    raise BuildError("mojo compiler not found")


def build(force: bool = False) -> str:
    if os.environ.get("MOJO_VOWPALWABBIT_LIB") and os.path.exists(LIB) and not force:
        return LIB
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(SRC):
        return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    cmd = _mojo_command() + ["build", "--emit", "shared-lib", SRC, "-o", LIB]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_LIBRARY: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_LIBRARY, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _LIBRARY


def addr(array: np.ndarray) -> int:
    return int(array.ctypes.data)


def bytes_array(value: str | bytes) -> tuple[np.ndarray, int]:
    encoded = value.encode("utf-8") if isinstance(value, str) else value
    storage = np.frombuffer(encoded + b"\0", dtype=np.uint8)
    return storage, len(encoded)
