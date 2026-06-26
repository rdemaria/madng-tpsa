"""Python wrapper for MAD-NG real TPSA values and maps.

Importing this package is cheap and does not load MAD-NG. The C library is
opened lazily when you build a descriptor or explicitly call ``load_library``.
"""

from __future__ import annotations

from ._cffi import availability_error, is_available, load_library, loaded_library_path
from .descriptor import Descriptor, DescriptorBuilder, descriptor
from .exceptions import DescriptorClosedError, MadngLibraryError, MadngTPSAError, TPSAClosedError
from .map import TPSAMap
from .tpsa import TPSA, from_mapping
from . import functions
from .functions import (
    abs,
    acos,
    acosh,
    acot,
    acoth,
    asin,
    asinc,
    asinh,
    asinhc,
    atan,
    atan2,
    atanh,
    cos,
    cosh,
    cot,
    coth,
    erf,
    erfc,
    erfcx,
    erfi,
    exp,
    hypot,
    hypot3,
    invsqrt,
    log,
    sin,
    sincos,
    sincosh,
    sinc,
    sinh,
    sinhc,
    sqrt,
    tan,
    tanh,
    unit,
    wf,
)

__version__ = "0.3.1"

__all__ = [
    "Descriptor",
    "DescriptorBuilder",
    "TPSA",
    "TPSAMap",
    "descriptor",
    "from_mapping",
    "load_library",
    "is_available",
    "availability_error",
    "loaded_library_path",
    "MadngTPSAError",
    "MadngLibraryError",
    "DescriptorClosedError",
    "TPSAClosedError",
    "functions",
    "abs",
    "acos",
    "acosh",
    "acot",
    "acoth",
    "asin",
    "asinc",
    "asinh",
    "asinhc",
    "atan",
    "atan2",
    "atanh",
    "cos",
    "cosh",
    "cot",
    "coth",
    "erf",
    "erfc",
    "erfcx",
    "erfi",
    "exp",
    "hypot",
    "hypot3",
    "invsqrt",
    "log",
    "sin",
    "sincos",
    "sincosh",
    "sinc",
    "sinh",
    "sinhc",
    "sqrt",
    "tan",
    "tanh",
    "unit",
    "wf",
]
