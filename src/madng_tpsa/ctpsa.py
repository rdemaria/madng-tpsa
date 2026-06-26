"""Pythonic complex TPSA value wrapper."""

from __future__ import annotations

import cmath
from numbers import Complex, Real
from typing import Iterator, Mapping, Sequence

from ._cffi import default_order, ffi, load_library, same_order, _ord, ptr_address
from .descriptor import Descriptor, _monomial_array
from .exceptions import MadngTPSAError, TPSAClosedError
from .tpsa import TPSA

Scalar = int | float | complex


def _is_scalar(value: object) -> bool:
    return isinstance(value, Complex) and not isinstance(value, bool)


def _parts(value: complex | float | int) -> tuple[float, float]:
    z = complex(value)
    return float(z.real), float(z.imag)


class CTPSA:
    """A complex MAD-NG truncated power-series value.

    ``CTPSA`` wraps the ``mad_ctpsa_*`` C API.  The Python layer uses the
    ``*_r`` entry points for scalar complex arguments, so complex values cross
    the CFFI boundary as explicit real/imaginary double pairs.
    """

    __array_priority__ = 1001
    __slots__ = ("_ptr", "_desc", "_owns", "_closed")

    def __init__(
        self,
        descriptor: Descriptor,
        value: Scalar = 0.0,
        *,
        order: int | None = None,
        _ptr=None,
        _owns: bool = True,
    ) -> None:
        descriptor._assert_open()
        self._desc = descriptor
        self._owns = _owns
        self._closed = False
        if _ptr is None:
            lib = load_library()
            ord_value = default_order() if order is None else _ord(int(order))
            ptr = lib.mad_ctpsa_newd(descriptor.ptr, ord_value)
            if ptr == ffi.NULL:
                raise MadngTPSAError("MAD-NG returned NULL from mad_ctpsa_newd")
            self._ptr = ptr
            re, im = _parts(value)
            lib.mad_ctpsa_setval_r(self._ptr, re, im)
        else:
            if _ptr == ffi.NULL:
                raise MadngTPSAError("cannot wrap NULL ctpsa_t pointer")
            self._ptr = _ptr

    @classmethod
    def _from_ptr(cls, descriptor: Descriptor, ptr, *, owns: bool = True) -> "CTPSA":
        return cls(descriptor, _ptr=ptr, _owns=owns)

    @classmethod
    def constant(cls, descriptor: Descriptor, value: Scalar = 0.0, *, order: int | None = None) -> "CTPSA":
        return cls(descriptor, value, order=order)

    @classmethod
    def variable(
        cls,
        descriptor: Descriptor,
        index: int,
        *,
        value: Scalar = 0.0,
        scale: Scalar = 1.0,
        order: int | None = None,
    ) -> "CTPSA":
        if not (1 <= int(index) <= descriptor.n_variables):
            raise IndexError(f"variable index must be in 1..{descriptor.n_variables}, got {index}")
        out = cls(descriptor, 0.0, order=order)
        v_re, v_im = _parts(value)
        s_re, s_im = _parts(scale)
        load_library().mad_ctpsa_setvar_r(out._ptr, v_re, v_im, int(index), s_re, s_im)
        return out

    @classmethod
    def parameter(
        cls,
        descriptor: Descriptor,
        index: int,
        *,
        value: Scalar = 0.0,
        order: int | None = None,
    ) -> "CTPSA":
        if not (1 <= int(index) <= descriptor.n_parameters):
            raise IndexError(f"parameter index must be in 1..{descriptor.n_parameters}, got {index}")
        out = cls(descriptor, 0.0, order=order if order is not None else 1)
        v_re, v_im = _parts(value)
        load_library().mad_ctpsa_setprm_r(out._ptr, v_re, v_im, int(index))
        return out

    @classmethod
    def from_tpsa(cls, real: TPSA, imag: TPSA | None = None) -> "CTPSA":
        real._assert_open()
        if imag is not None:
            imag._assert_open()
            real._assert_compatible(imag)
        out = cls(real.descriptor, 0.0, order=real.order)
        load_library().mad_ctpsa_cplx(real.ptr, imag.ptr if imag is not None else ffi.NULL, out.ptr)
        return out

    @property
    def descriptor(self) -> Descriptor:
        return self._desc

    @property
    def ptr(self):
        self._assert_open()
        return self._ptr

    @property
    def address(self) -> int:
        self._assert_open()
        return ptr_address(self._ptr)

    @property
    def order(self) -> int:
        self._assert_open()
        return int(load_library().mad_ctpsa_ord(self._ptr, False))

    @property
    def high_order(self) -> int:
        self._assert_open()
        return int(load_library().mad_ctpsa_ord(self._ptr, True))

    @property
    def length(self) -> int:
        self._assert_open()
        return int(load_library().mad_ctpsa_len(self._ptr, False))

    @property
    def constant_term(self) -> complex:
        return complex(self[()])

    @property
    def real(self) -> TPSA:
        out = TPSA.constant(self.descriptor, 0.0, order=self.order)
        load_library().mad_ctpsa_real(self.ptr, out.ptr)
        return out

    @property
    def imag(self) -> TPSA:
        out = TPSA.constant(self.descriptor, 0.0, order=self.order)
        load_library().mad_ctpsa_imag(self.ptr, out.ptr)
        return out

    def abs(self) -> TPSA:
        out = TPSA.constant(self.descriptor, 0.0, order=self.order)
        load_library().mad_ctpsa_cabs(self.ptr, out.ptr)
        return out

    def arg(self) -> TPSA:
        out = TPSA.constant(self.descriptor, 0.0, order=self.order)
        load_library().mad_ctpsa_carg(self.ptr, out.ptr)
        return out

    def copy(self, *, order: int | None = None) -> "CTPSA":
        self._assert_open()
        lib = load_library()
        ptr = lib.mad_ctpsa_new(self._ptr, same_order() if order is None else _ord(int(order)))
        out = CTPSA._from_ptr(self._desc, ptr)
        lib.mad_ctpsa_copy(self._ptr, out._ptr)
        return out

    def clear(self) -> "CTPSA":
        self._assert_open()
        load_library().mad_ctpsa_clear(self._ptr)
        return self

    def update(self) -> "CTPSA":
        self._assert_open()
        load_library().mad_ctpsa_update(self._ptr)
        return self

    def coefficient(self, monomial: Sequence[int] | int | str | tuple[()]) -> complex:
        return complex(self[monomial])

    def set_coefficient(self, monomial: Sequence[int] | int | str | tuple[()], value: Scalar) -> "CTPSA":
        self[monomial] = value
        return self

    def coefficients(self) -> Iterator[tuple[tuple[int, ...], complex]]:
        self._assert_open()
        lib = load_library()
        n = self.descriptor.n_total
        mono = ffi.new("ord_t[]", n)
        value = ffi.new("cpx_t *")
        i = -1
        while True:
            i = int(lib.mad_ctpsa_cycle(self._ptr, i, n, mono, value))
            if i < 0:
                break
            yield tuple(int(mono[j]) for j in range(n)), complex(value[0])

    def to_dict(self) -> dict[tuple[int, ...], complex]:
        return dict(self.coefficients())

    def get_order(self, order: int) -> "CTPSA":
        self._assert_open()
        out = self._new_like()
        load_library().mad_ctpsa_getord(self._ptr, out._ptr, _ord(int(order)))
        return out

    def cut_order(self, order: int) -> "CTPSA":
        self._assert_open()
        out = self._new_like()
        load_library().mad_ctpsa_cutord(self._ptr, out._ptr, int(order))
        return out

    def clear_order(self, order: int) -> "CTPSA":
        self._assert_open()
        load_library().mad_ctpsa_clrord(self._ptr, _ord(int(order)))
        return self

    def derivative(self, variable: int) -> "CTPSA":
        self._assert_open()
        out = self._new_like()
        load_library().mad_ctpsa_deriv(self._ptr, out._ptr, int(variable))
        return out

    def integrate(self, variable: int) -> "CTPSA":
        self._assert_open()
        out = self._new_like()
        load_library().mad_ctpsa_integ(self._ptr, out._ptr, int(variable))
        return out

    def poisson_bracket(self, other: "CTPSA | TPSA | Scalar", *, n_variables: int | None = None) -> "CTPSA":
        other = self._coerce_ctpsa(other)
        out = self._new_like()
        nv = self.descriptor.n_variables if n_variables is None else int(n_variables)
        load_library().mad_ctpsa_poisbra(self._ptr, other._ptr, out._ptr, nv)
        return out

    def evaluate(self, values: Sequence[complex | float]) -> complex:
        from .ctpsa_map import CTPSAMap

        return CTPSAMap([self]).evaluate(values)[0]

    def compose(self, substitutions: "Sequence[CTPSA | TPSA] | object") -> "CTPSA":
        from .ctpsa_map import CTPSAMap

        sub_map = substitutions if isinstance(substitutions, CTPSAMap) else CTPSAMap(substitutions)  # type: ignore[arg-type]
        return CTPSAMap([self]).compose(sub_map)[0]

    def norm(self) -> float:
        self._assert_open()
        return float(load_library().mad_ctpsa_nrm(self._ptr))

    def almost_equal(self, other: "CTPSA | TPSA | Scalar", *, tol: float = 0.0) -> bool:
        other = self._coerce_ctpsa(other)
        return bool(load_library().mad_ctpsa_equ(self._ptr, other._ptr, float(tol)))

    def is_null(self) -> bool:
        self._assert_open()
        return bool(load_library().mad_ctpsa_isnul(self._ptr))

    def is_scalar(self) -> bool:
        self._assert_open()
        return bool(load_library().mad_ctpsa_isval(self._ptr))

    def is_valid(self) -> bool:
        self._assert_open()
        return bool(load_library().mad_ctpsa_isvalid(self._ptr))

    def _new_like(self) -> "CTPSA":
        self._assert_open()
        ptr = load_library().mad_ctpsa_new(self._ptr, same_order())
        return CTPSA._from_ptr(self._desc, ptr)

    def _unary(self, c_function_name: str) -> "CTPSA":
        out = self._new_like()
        getattr(load_library(), c_function_name)(self._ptr, out._ptr)
        return out

    def _binary(self, other: "CTPSA | TPSA | Scalar", c_function_name: str) -> "CTPSA":
        other = self._coerce_ctpsa(other)
        out = self._new_like()
        getattr(load_library(), c_function_name)(self._ptr, other._ptr, out._ptr)
        return out

    def _coerce_ctpsa(self, other: "CTPSA | TPSA | Scalar") -> "CTPSA":
        if isinstance(other, CTPSA):
            other._assert_open()
            self._assert_compatible(other)
            return other
        if isinstance(other, TPSA):
            other._assert_open()
            if self.descriptor.address != other.descriptor.address:
                raise ValueError("TPSA descriptors differ")
            return CTPSA.from_tpsa(other)
        if _is_scalar(other):
            return CTPSA.constant(self.descriptor, complex(other), order=self.order)
        return NotImplemented  # type: ignore[return-value]

    def _assert_compatible(self, other: "CTPSA") -> None:
        if self.descriptor.address != other.descriptor.address:
            raise ValueError("CTPSA descriptors differ")

    def _assert_open(self) -> None:
        if self._closed or self._ptr == ffi.NULL:
            raise TPSAClosedError("complex TPSA value has been closed")
        self._desc._assert_open()

    def close(self) -> None:
        if not self._closed and self._owns:
            load_library().mad_ctpsa_del(self._ptr)
        self._closed = True
        self._ptr = ffi.NULL

    def __getitem__(self, monomial: Sequence[int] | int | str | tuple[()]) -> complex:
        self._assert_open()
        lib = load_library()
        value = ffi.new("cpx_t *")
        if monomial == ():
            lib.mad_ctpsa_geti_r(self._ptr, 0, value)
            return complex(value[0])
        if isinstance(monomial, int):
            if monomial < 0:
                raise IndexError("coefficient index must be non-negative")
            lib.mad_ctpsa_geti_r(self._ptr, int(monomial), value)
            return complex(value[0])
        if isinstance(monomial, str):
            values = tuple(int(ch, 36) for ch in monomial.strip())
        else:
            values = tuple(int(v) for v in monomial)
        m = _monomial_array(values, self.descriptor.n_total)
        lib.mad_ctpsa_getm_r(self._ptr, self.descriptor.n_total, m, value)
        return complex(value[0])

    def __setitem__(self, monomial: Sequence[int] | int | str | tuple[()], value: Scalar) -> None:
        self._assert_open()
        lib = load_library()
        re, im = _parts(value)
        if monomial == ():
            lib.mad_ctpsa_seti_r(self._ptr, 0, 0.0, 0.0, re, im)
            return
        if isinstance(monomial, int):
            if monomial < 0:
                raise IndexError("coefficient index must be non-negative")
            lib.mad_ctpsa_seti_r(self._ptr, int(monomial), 0.0, 0.0, re, im)
            return
        if isinstance(monomial, str):
            values = tuple(int(ch, 36) for ch in monomial.strip())
        else:
            values = tuple(int(v) for v in monomial)
        m = _monomial_array(values, self.descriptor.n_total)
        lib.mad_ctpsa_setm_r(self._ptr, self.descriptor.n_total, m, 0.0, 0.0, re, im)

    def __add__(self, other: "CTPSA | TPSA | Scalar"):
        if _is_scalar(other):
            out = self.copy()
            re, im = _parts(other)
            load_library().mad_ctpsa_seti_r(out._ptr, 0, 1.0, 0.0, re, im)
            return out
        if isinstance(other, (CTPSA, TPSA)):
            return self._binary(other, "mad_ctpsa_add")
        return NotImplemented

    def __radd__(self, other: Scalar):
        return self.__add__(other)

    def __sub__(self, other: "CTPSA | TPSA | Scalar"):
        if _is_scalar(other):
            out = self.copy()
            re, im = _parts(other)
            load_library().mad_ctpsa_seti_r(out._ptr, 0, 1.0, 0.0, -re, -im)
            return out
        if isinstance(other, (CTPSA, TPSA)):
            return self._binary(other, "mad_ctpsa_sub")
        return NotImplemented

    def __rsub__(self, other: Scalar):
        if _is_scalar(other):
            return (-self).__add__(other)
        return NotImplemented

    def __mul__(self, other: "CTPSA | TPSA | Scalar"):
        if _is_scalar(other):
            out = self._new_like()
            re, im = _parts(other)
            load_library().mad_ctpsa_scl_r(self._ptr, re, im, out._ptr)
            return out
        if isinstance(other, (CTPSA, TPSA)):
            return self._binary(other, "mad_ctpsa_mul")
        return NotImplemented

    def __rmul__(self, other: Scalar):
        return self.__mul__(other)

    def __truediv__(self, other: "CTPSA | TPSA | Scalar"):
        if _is_scalar(other):
            out = self._new_like()
            re, im = _parts(other)
            load_library().mad_ctpsa_divn_r(self._ptr, re, im, out._ptr)
            return out
        if isinstance(other, (CTPSA, TPSA)):
            return self._binary(other, "mad_ctpsa_div")
        return NotImplemented

    def __rtruediv__(self, other: Scalar):
        if _is_scalar(other):
            out = self._new_like()
            re, im = _parts(other)
            load_library().mad_ctpsa_inv_r(self._ptr, re, im, out._ptr)
            return out
        return NotImplemented

    def __pow__(self, other: "CTPSA | TPSA | Scalar"):
        self._assert_open()
        out = self._new_like()
        lib = load_library()
        if isinstance(other, bool):
            other = int(other)
        if isinstance(other, int):
            lib.mad_ctpsa_powi(self._ptr, int(other), out._ptr)
            return out
        if isinstance(other, Real) or isinstance(other, complex):
            re, im = _parts(other)
            lib.mad_ctpsa_pown_r(self._ptr, re, im, out._ptr)
            return out
        if isinstance(other, (CTPSA, TPSA)):
            other_c = self._coerce_ctpsa(other)
            lib.mad_ctpsa_pow(self._ptr, other_c._ptr, out._ptr)
            return out
        return NotImplemented

    def __rpow__(self, other: Scalar):
        if _is_scalar(other):
            from .functions import exp

            return exp(cmath.log(complex(other)) * self)
        return NotImplemented

    def __neg__(self) -> "CTPSA":
        out = self._new_like()
        load_library().mad_ctpsa_scl_r(self._ptr, -1.0, 0.0, out._ptr)
        return out

    def __pos__(self) -> "CTPSA":
        return self.copy()

    def __abs__(self) -> TPSA:
        return self.abs()

    def conjugate(self) -> "CTPSA":
        return self._unary("mad_ctpsa_conj")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (CTPSA, TPSA)) or _is_scalar(other):
            try:
                return self.almost_equal(other)  # type: ignore[arg-type]
            except Exception:
                return False
        return False

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __complex__(self) -> complex:
        return self.constant_term

    def __iter__(self):
        return self.coefficients()

    def __enter__(self) -> "CTPSA":
        self._assert_open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - destructor timing is interpreter-specific
        try:
            if getattr(self, "_owns", False) and not getattr(self, "_closed", True):
                lib = load_library()
                lib.mad_ctpsa_del(self._ptr)
                self._closed = True
        except Exception:
            pass

    def __repr__(self) -> str:
        if self._closed:
            return "CTPSA(closed=True)"
        terms = list(self.coefficients())
        shown = ", ".join(f"{m}: {v:.6g}" for m, v in terms[:6])
        if len(terms) > 6:
            shown += ", ..."
        return f"CTPSA({{{shown}}}, order={self.order}, descriptor=0x{self.descriptor.address:x})"


def coerce_common_complex(a: CTPSA | TPSA | Scalar, b: CTPSA | TPSA | Scalar) -> tuple[CTPSA, CTPSA]:
    if isinstance(a, CTPSA):
        return a, a._coerce_ctpsa(b)
    if isinstance(b, CTPSA):
        return b._coerce_ctpsa(a), b
    if isinstance(a, TPSA) and isinstance(b, TPSA):
        a._assert_compatible(b)
        return CTPSA.from_tpsa(a), CTPSA.from_tpsa(b)
    if isinstance(a, TPSA) and _is_scalar(b):
        return CTPSA.from_tpsa(a), CTPSA.constant(a.descriptor, b, order=a.order)
    if isinstance(b, TPSA) and _is_scalar(a):
        return CTPSA.constant(b.descriptor, a, order=b.order), CTPSA.from_tpsa(b)
    raise TypeError("at least one operand must be a TPSA or CTPSA value")


def complex_from_mapping(descriptor: Descriptor, terms: Mapping[Sequence[int], Scalar], *, order: int | None = None) -> CTPSA:
    """Create a complex TPSA from a mapping of monomial tuples to coefficients."""

    out = CTPSA.constant(descriptor, 0.0, order=order)
    for monomial, value in terms.items():
        out[monomial] = value
    return out
        """Create a complex constant TPSA value."""

        """Create a one-based complex TPSA variable."""

        """Create a one-based complex TPSA parameter."""

        """Promote real and optional imaginary TPSA values to one complex TPSA."""

        """Return the descriptor used by this complex TPSA value."""

        """Return the underlying ctpsa_t CFFI pointer."""

        """Return the integer address of the wrapped ctpsa_t pointer."""

        """Return the allocated maximum Taylor order."""

        """Return the highest currently non-zero Taylor order."""

        """Return the allocated coefficient-vector length."""

        """Return the complex coefficient of the zero monomial."""

        """Return the real projection as a TPSA value."""

        """Return the imaginary projection as a TPSA value."""

        """Return the complex magnitude as a real TPSA value."""

        """Return the complex phase as a real TPSA value."""

        """Return a copy of this complex TPSA value."""

        """Clear all coefficients in place and return self."""

        """Refresh internal low/high order metadata after coefficient changes."""

        """Return a complex coefficient."""

        """Set one complex coefficient and return self."""

        """Yield non-zero complex coefficients as monomial-value pairs."""

        """Return all non-zero complex coefficients as a dictionary."""

        """Return the homogeneous component of the requested order."""

        """Return a copy with high or low orders removed according to MAD-NG semantics."""

        """Clear one homogeneous order in place and return self."""

        """Return the derivative with respect to a one-based variable index."""

        """Return the integral with respect to a one-based variable index."""

        """Return the complex Poisson bracket with another value."""

        """Evaluate this complex TPSA for a vector of values."""

        """Compose this complex TPSA with a substitution map."""

        """Return the MAD-NG complex TPSA norm."""

        """Compare two complex TPSA values using a coefficient tolerance."""

        """Return True when this complex TPSA is identically zero."""

        """Return True when this complex TPSA has no non-constant terms."""

        """Return True when the wrapped C CTPSA object passes validation."""

        """Release the wrapped ctpsa_t pointer if owned by this object."""

        """Return a complex coefficient by monomial tuple, string, or raw index."""

        """Set a complex coefficient by monomial tuple, string, or raw index."""

        """Return the complex TPSA sum with another compatible value."""

        """Return the reflected complex TPSA sum."""

        """Return the complex TPSA difference with another compatible value."""

        """Return the reflected complex TPSA difference."""

        """Return the complex TPSA product with another compatible value."""

        """Return the reflected complex TPSA product."""

        """Return the complex TPSA quotient by another compatible value."""

        """Return the reflected complex TPSA quotient."""

        """Raise this complex TPSA to a scalar or TPSA power."""

        """Raise a scalar to this complex TPSA power."""

        """Return the additive inverse of this complex TPSA."""

        """Return a copy of this complex TPSA."""

        """Return the complex magnitude as a real TPSA value."""

        """Return the complex conjugate of this TPSA."""

        """Return coefficient-wise equality with a compatible value."""

        """Return coefficient-wise inequality with a compatible value."""

        """Return the constant term as a Python complex."""

        """Iterate over non-zero complex coefficient pairs."""

        """Return this complex TPSA for use as a context manager."""

        """Close this complex TPSA when leaving a context-manager block."""

        """Return a concise representation with a few non-zero coefficients."""

    """Promote two operands to compatible CTPSA values."""

