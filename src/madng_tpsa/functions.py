"""Mathematical functions for TPSA values and maps."""

from __future__ import annotations

import cmath
import math
from numbers import Real
from typing import Callable, Union

from ._cffi import load_library
from .ctpsa import CTPSA, coerce_common_complex
from .ctpsa_map import CTPSAMap
from .map import TPSAMap
from .tpsa import TPSA, coerce_common


Number = Union[int, float]
ComplexNumber = Union[int, float, complex]


def _is_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _is_complex_number(value: object) -> bool:
    return isinstance(value, complex) or _is_number(value)


def _ct_name(c_function_name: str) -> str:
    return c_function_name.replace("mad_tpsa_", "mad_ctpsa_")


def _unary(
    value,
    c_function_name: str,
    numeric: Callable[[float], float] | None = None,
    cnumeric: Callable[[complex], complex] | None = None,
):
    if isinstance(value, CTPSA):
        return value._unary(_ct_name(c_function_name))
    if isinstance(value, CTPSAMap):
        return value.apply(lambda component: _unary(component, c_function_name, numeric, cnumeric))
    if isinstance(value, TPSA):
        return value._unary(c_function_name)
    if isinstance(value, TPSAMap):
        return value.apply(lambda component: _unary(component, c_function_name, numeric, cnumeric))
    if isinstance(value, complex) and cnumeric is not None:
        return cnumeric(value)
    if _is_number(value) and numeric is not None:
        return numeric(float(value))
    raise TypeError(f"{c_function_name[9:]} expects a TPSA, CTPSA, map, or supported number")


def _binary(a, b, c_function_name: str, numeric: Callable[[float, float], float] | None = None):
    if isinstance(a, CTPSAMap) and isinstance(b, CTPSAMap):
        if len(a) != len(b):
            raise ValueError("CTPSAMap sizes differ")
        return CTPSAMap(_binary(x, y, c_function_name, numeric) for x, y in zip(a, b))
    if isinstance(a, CTPSAMap):
        return CTPSAMap(_binary(x, b, c_function_name, numeric) for x in a)
    if isinstance(b, CTPSAMap):
        return CTPSAMap(_binary(a, y, c_function_name, numeric) for y in b)
    if isinstance(a, TPSAMap) and isinstance(b, TPSAMap):
        if len(a) != len(b):
            raise ValueError("TPSAMap sizes differ")
        return TPSAMap(_binary(x, y, c_function_name, numeric) for x, y in zip(a, b))
    if isinstance(a, TPSAMap):
        return TPSAMap(_binary(x, b, c_function_name, numeric) for x in a)
    if isinstance(b, TPSAMap):
        return TPSAMap(_binary(a, y, c_function_name, numeric) for y in b)
    if isinstance(a, CTPSA) or isinstance(b, CTPSA):
        aa, bb = coerce_common_complex(a, b)
        out = aa._new_like()
        getattr(load_library(), _ct_name(c_function_name))(aa.ptr, bb.ptr, out.ptr)
        return out
    if isinstance(a, TPSA) or isinstance(b, TPSA):
        aa, bb = coerce_common(a, b)
        out = aa._new_like()
        getattr(load_library(), c_function_name)(aa.ptr, bb.ptr, out.ptr)
        return out
    if _is_number(a) and _is_number(b) and numeric is not None:
        return numeric(float(a), float(b))
    raise TypeError(f"{c_function_name[9:]} expects TPSA-compatible arguments")


def unit(x):
    return _unary(
        x,
        "mad_tpsa_unit",
        lambda v: math.copysign(1.0, v) if v else 0.0,
        lambda z: z / builtins_abs(z) if z else 0j,
    )


def abs(x):  # noqa: A001 - intentionally mirrors built-in for TPSA math namespace
    if isinstance(x, CTPSA):
        return x.abs()
    if isinstance(x, CTPSAMap):
        return TPSAMap(xi.abs() for xi in x)
    return _unary(x, "mad_tpsa_abs", math.fabs, builtins_abs)


def builtins_abs(z: complex) -> float:
    import builtins

    return builtins.abs(z)


def conj(x):
    if isinstance(x, CTPSA):
        return x.conjugate()
    if isinstance(x, CTPSAMap):
        return x.apply(lambda component: component.conjugate())
    if isinstance(x, TPSA):
        return x.copy()
    if isinstance(x, TPSAMap):
        return x.copy()
    if _is_complex_number(x):
        return complex(x).conjugate()
    raise TypeError("conj expects TPSA-compatible arguments")


def real(x):
    if isinstance(x, CTPSA):
        return x.real
    if isinstance(x, CTPSAMap):
        return TPSAMap(component.real for component in x)
    if isinstance(x, TPSA):
        return x
    if isinstance(x, TPSAMap):
        return x
    if _is_complex_number(x):
        return complex(x).real
    raise TypeError("real expects TPSA-compatible arguments")


def imag(x):
    if isinstance(x, CTPSA):
        return x.imag
    if isinstance(x, CTPSAMap):
        return TPSAMap(component.imag for component in x)
    if isinstance(x, TPSA):
        return TPSA.constant(x.descriptor, 0.0, order=x.order)
    if isinstance(x, TPSAMap):
        return TPSAMap(TPSA.constant(component.descriptor, 0.0, order=component.order) for component in x)
    if _is_complex_number(x):
        return complex(x).imag
    raise TypeError("imag expects TPSA-compatible arguments")


def carg(x):
    if isinstance(x, CTPSA):
        return x.arg()
    if isinstance(x, CTPSAMap):
        return TPSAMap(component.arg() for component in x)
    if _is_complex_number(x):
        return cmath.phase(complex(x))
    raise TypeError("carg expects CTPSA-compatible arguments")


def sqrt(x):
    return _unary(x, "mad_tpsa_sqrt", math.sqrt, cmath.sqrt)


def exp(x):
    return _unary(x, "mad_tpsa_exp", math.exp, cmath.exp)


def log(x):
    return _unary(x, "mad_tpsa_log", math.log, cmath.log)


def sin(x):
    return _unary(x, "mad_tpsa_sin", math.sin, cmath.sin)


def cos(x):
    return _unary(x, "mad_tpsa_cos", math.cos, cmath.cos)


def tan(x):
    return _unary(x, "mad_tpsa_tan", math.tan, cmath.tan)


def cot(x):
    return _unary(x, "mad_tpsa_cot", lambda v: 1.0 / math.tan(v))


def sinc(x):
    return _unary(
        x,
        "mad_tpsa_sinc",
        lambda v: 1.0 if v == 0.0 else math.sin(v) / v,
        lambda z: 1.0 if z == 0 else cmath.sin(z) / z,
    )


def sinh(x):
    return _unary(x, "mad_tpsa_sinh", math.sinh, cmath.sinh)


def cosh(x):
    return _unary(x, "mad_tpsa_cosh", math.cosh, cmath.cosh)


def tanh(x):
    return _unary(x, "mad_tpsa_tanh", math.tanh, cmath.tanh)


def coth(x):
    return _unary(x, "mad_tpsa_coth", lambda v: 1.0 / math.tanh(v))


def sinhc(x):
    return _unary(
        x,
        "mad_tpsa_sinhc",
        lambda v: 1.0 if v == 0.0 else math.sinh(v) / v,
        lambda z: 1.0 if z == 0 else cmath.sinh(z) / z,
    )


def asin(x):
    return _unary(x, "mad_tpsa_asin", math.asin)


def acos(x):
    return _unary(x, "mad_tpsa_acos", math.acos)


def atan(x):
    return _unary(x, "mad_tpsa_atan", math.atan)


def acot(x):
    return _unary(x, "mad_tpsa_acot", lambda v: math.atan(1.0 / v))


def asinc(x):
    return _unary(x, "mad_tpsa_asinc", None)


def asinh(x):
    return _unary(x, "mad_tpsa_asinh", math.asinh)


def acosh(x):
    return _unary(x, "mad_tpsa_acosh", math.acosh)


def atanh(x):
    return _unary(x, "mad_tpsa_atanh", math.atanh)


def acoth(x):
    return _unary(x, "mad_tpsa_acoth", lambda v: math.atanh(1.0 / v))


def asinhc(x):
    return _unary(x, "mad_tpsa_asinhc", None)


def erf(x):
    return _unary(x, "mad_tpsa_erf", math.erf)


def erfc(x):
    return _unary(x, "mad_tpsa_erfc", math.erfc)


def erfcx(x):
    return _unary(x, "mad_tpsa_erfcx", lambda v: math.exp(v * v) * math.erfc(v))


def erfi(x):
    return _unary(x, "mad_tpsa_erfi", None)


def wf(x):
    return _unary(x, "mad_tpsa_wf", None)


def invsqrt(x, scale: complex | float = 1.0):
    if isinstance(x, CTPSA):
        out = x._new_like()
        z = complex(scale)
        load_library().mad_ctpsa_invsqrt_r(x.ptr, z.real, z.imag, out.ptr)
        return out
    if isinstance(x, CTPSAMap):
        return x.apply(lambda component: invsqrt(component, scale))
    if isinstance(x, TPSA):
        if isinstance(scale, complex) and scale.imag:
            return invsqrt(CTPSA.from_tpsa(x), scale)
        out = x._new_like()
        load_library().mad_tpsa_invsqrt(x.ptr, float(scale), out.ptr)
        return out
    if isinstance(x, TPSAMap):
        return x.apply(lambda component: invsqrt(component, scale))
    if isinstance(x, complex):
        return complex(scale) / cmath.sqrt(x)
    if _is_number(x):
        return float(scale) / math.sqrt(float(x))
    raise TypeError("invsqrt expects a TPSA, CTPSA, map, or number")


def sincos(x):
    if isinstance(x, CTPSA):
        s = x._new_like()
        c = x._new_like()
        load_library().mad_ctpsa_sincos(x.ptr, s.ptr, c.ptr)
        return s, c
    if isinstance(x, CTPSAMap):
        pairs = [sincos(component) for component in x]
        return CTPSAMap(s for s, _ in pairs), CTPSAMap(c for _, c in pairs)
    if isinstance(x, TPSA):
        s = x._new_like()
        c = x._new_like()
        load_library().mad_tpsa_sincos(x.ptr, s.ptr, c.ptr)
        return s, c
    if isinstance(x, TPSAMap):
        pairs = [sincos(component) for component in x]
        return TPSAMap(s for s, _ in pairs), TPSAMap(c for _, c in pairs)
    if isinstance(x, complex):
        return cmath.sin(x), cmath.cos(x)
    if _is_number(x):
        return math.sin(float(x)), math.cos(float(x))
    raise TypeError("sincos expects a TPSA, CTPSA, map, or number")


def sincosh(x):
    if isinstance(x, CTPSA):
        s = x._new_like()
        c = x._new_like()
        load_library().mad_ctpsa_sincosh(x.ptr, s.ptr, c.ptr)
        return s, c
    if isinstance(x, CTPSAMap):
        pairs = [sincosh(component) for component in x]
        return CTPSAMap(s for s, _ in pairs), CTPSAMap(c for _, c in pairs)
    if isinstance(x, TPSA):
        s = x._new_like()
        c = x._new_like()
        load_library().mad_tpsa_sincosh(x.ptr, s.ptr, c.ptr)
        return s, c
    if isinstance(x, TPSAMap):
        pairs = [sincosh(component) for component in x]
        return TPSAMap(s for s, _ in pairs), TPSAMap(c for _, c in pairs)
    if isinstance(x, complex):
        return cmath.sinh(x), cmath.cosh(x)
    if _is_number(x):
        return math.sinh(float(x)), math.cosh(float(x))
    raise TypeError("sincosh expects a TPSA, CTPSA, map, or number")


def atan2(y, x):
    return _binary(y, x, "mad_tpsa_atan2", math.atan2)


def hypot(x, y):
    return _binary(x, y, "mad_tpsa_hypot", math.hypot)


def hypot3(x, y, z):
    if isinstance(x, CTPSAMap) or isinstance(y, CTPSAMap) or isinstance(z, CTPSAMap):
        maps = [arg for arg in (x, y, z) if isinstance(arg, CTPSAMap)]
        size = len(maps[0])
        if any(len(m) != size for m in maps):
            raise ValueError("CTPSAMap sizes differ")
        return CTPSAMap(hypot3(_component_or_scalar(x, i), _component_or_scalar(y, i), _component_or_scalar(z, i)) for i in range(size))
    if isinstance(x, CTPSA) or isinstance(y, CTPSA) or isinstance(z, CTPSA):
        a, b = coerce_common_complex(x, y)
        a, c = coerce_common_complex(a, z)
        out = a._new_like()
        load_library().mad_ctpsa_hypot3(a.ptr, b.ptr, c.ptr, out.ptr)
        return out
    if isinstance(x, TPSAMap) or isinstance(y, TPSAMap) or isinstance(z, TPSAMap):
        maps = [arg for arg in (x, y, z) if isinstance(arg, TPSAMap)]
        size = len(maps[0])
        if any(len(m) != size for m in maps):
            raise ValueError("TPSAMap sizes differ")
        return TPSAMap(hypot3(_component_or_scalar(x, i), _component_or_scalar(y, i), _component_or_scalar(z, i)) for i in range(size))
    if isinstance(x, TPSA) or isinstance(y, TPSA) or isinstance(z, TPSA):
        a, b = coerce_common(x, y)
        a, c = coerce_common(a, z)
        out = a._new_like()
        load_library().mad_tpsa_hypot3(a.ptr, b.ptr, c.ptr, out.ptr)
        return out
    if _is_number(x) and _is_number(y) and _is_number(z):
        return math.sqrt(float(x) * float(x) + float(y) * float(y) + float(z) * float(z))
    raise TypeError("hypot3 expects TPSA-compatible arguments")


def _component_or_scalar(value, index: int):
    return value[index] if isinstance(value, (TPSAMap, CTPSAMap)) else value
