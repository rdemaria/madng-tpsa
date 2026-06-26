"""Pythonic TPSA value wrapper."""

from __future__ import annotations

import math as _math
from numbers import Real
from typing import Iterator, Mapping, Sequence, Union

from ._cffi import default_order, ffi, load_library, same_order, _ord, ptr_address
from .descriptor import Descriptor, _monomial_array
from .exceptions import MadngTPSAError, TPSAClosedError

Number = Union[int, float]


class TPSA:
    """A real MAD-NG truncated power-series value.

    ``TPSA`` supports Python algebra operators. Scalars are promoted to constant
    TPSA values using the left-hand operand's descriptor.
    """

    __array_priority__ = 1000
    __slots__ = ("_ptr", "_desc", "_owns", "_closed")

    def __init__(
        self,
        descriptor: Descriptor,
        value: float = 0.0,
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
            ptr = lib.mad_tpsa_newd(descriptor.ptr, ord_value)
            if ptr == ffi.NULL:
                raise MadngTPSAError("MAD-NG returned NULL from mad_tpsa_newd")
            self._ptr = ptr
            lib.mad_tpsa_setval(self._ptr, float(value))
        else:
            if _ptr == ffi.NULL:
                raise MadngTPSAError("cannot wrap NULL tpsa_t pointer")
            self._ptr = _ptr

    @classmethod
    def _from_ptr(cls, descriptor: Descriptor, ptr, *, owns: bool = True) -> "TPSA":
        return cls(descriptor, _ptr=ptr, _owns=owns)

    @classmethod
    def constant(cls, descriptor: Descriptor, value: float = 0.0, *, order: int | None = None) -> "TPSA":
        return cls(descriptor, value, order=order)

    @classmethod
    def variable(
        cls,
        descriptor: Descriptor,
        index: int,
        *,
        value: float = 0.0,
        scale: float = 1.0,
        order: int | None = None,
    ) -> "TPSA":
        if not (1 <= int(index) <= descriptor.n_variables):
            raise IndexError(f"variable index must be in 1..{descriptor.n_variables}, got {index}")
        out = cls(descriptor, 0.0, order=order)
        load_library().mad_tpsa_setvar(out._ptr, float(value), int(index), float(scale))
        return out

    @classmethod
    def parameter(
        cls,
        descriptor: Descriptor,
        index: int,
        *,
        value: float = 0.0,
        order: int | None = None,
    ) -> "TPSA":
        if not (1 <= int(index) <= descriptor.n_parameters):
            raise IndexError(f"parameter index must be in 1..{descriptor.n_parameters}, got {index}")
        out = cls(descriptor, 0.0, order=order if order is not None else 1)
        load_library().mad_tpsa_setprm(out._ptr, float(value), int(index))
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
        return int(load_library().mad_tpsa_ord(self._ptr, False))

    @property
    def high_order(self) -> int:
        self._assert_open()
        return int(load_library().mad_tpsa_ord(self._ptr, True))

    @property
    def length(self) -> int:
        self._assert_open()
        return int(load_library().mad_tpsa_len(self._ptr, False))

    @property
    def constant_term(self) -> float:
        return float(self[()])

    def copy(self, *, order: int | None = None) -> "TPSA":
        """Return a copy of this TPSA value."""

        self._assert_open()
        lib = load_library()
        if order is None:
            ptr = lib.mad_tpsa_new(self._ptr, same_order())
        else:
            ptr = lib.mad_tpsa_new(self._ptr, _ord(int(order)))
        out = TPSA._from_ptr(self._desc, ptr)
        lib.mad_tpsa_copy(self._ptr, out._ptr)
        return out

    def clear(self) -> "TPSA":
        self._assert_open()
        load_library().mad_tpsa_clear(self._ptr)
        return self

    def update(self) -> "TPSA":
        self._assert_open()
        load_library().mad_tpsa_update(self._ptr)
        return self

    def coefficient(self, monomial: Sequence[int] | int | str | tuple[()]) -> float:
        """Return a coefficient.

        ``monomial`` may be ``()`` for the constant term, a dense tuple such as
        ``(2, 0, 1)``, or a raw MAD-NG coefficient index as an integer.
        """

        return float(self[monomial])

    def set_coefficient(self, monomial: Sequence[int] | int | str | tuple[()], value: float) -> "TPSA":
        self[monomial] = value
        return self

    def coefficients(self) -> Iterator[tuple[tuple[int, ...], float]]:
        """Yield non-zero coefficients as ``(monomial, value)`` pairs."""

        self._assert_open()
        lib = load_library()
        n = self.descriptor.n_total
        mono = ffi.new("ord_t[]", n)
        value = ffi.new("num_t *")
        i = -1
        while True:
            i = int(lib.mad_tpsa_cycle(self._ptr, i, n, mono, value))
            if i < 0:
                break
            yield tuple(int(mono[j]) for j in range(n)), float(value[0])

    def to_dict(self) -> dict[tuple[int, ...], float]:
        return dict(self.coefficients())

    def get_order(self, order: int) -> "TPSA":
        self._assert_open()
        out = self._new_like()
        load_library().mad_tpsa_getord(self._ptr, out._ptr, _ord(int(order)))
        return out

    def cut_order(self, order: int) -> "TPSA":
        self._assert_open()
        out = self._new_like()
        load_library().mad_tpsa_cutord(self._ptr, out._ptr, int(order))
        return out

    def clear_order(self, order: int) -> "TPSA":
        self._assert_open()
        load_library().mad_tpsa_clrord(self._ptr, _ord(int(order)))
        return self

    def derivative(self, variable: int) -> "TPSA":
        self._assert_open()
        out = self._new_like()
        load_library().mad_tpsa_deriv(self._ptr, out._ptr, int(variable))
        return out

    def integrate(self, variable: int) -> "TPSA":
        self._assert_open()
        out = self._new_like()
        load_library().mad_tpsa_integ(self._ptr, out._ptr, int(variable))
        return out

    def poisson_bracket(self, other: "TPSA", *, n_variables: int | None = None) -> "TPSA":
        other = self._coerce_tpsa(other)
        out = self._new_like()
        nv = self.descriptor.n_variables if n_variables is None else int(n_variables)
        load_library().mad_tpsa_poisbra(self._ptr, other._ptr, out._ptr, nv)
        return out

    def evaluate(self, values: Sequence[float]) -> float:
        """Evaluate this TPSA for a vector of variable/parameter values."""

        from .map import TPSAMap

        return TPSAMap([self]).evaluate(values)[0]

    def compose(self, substitutions: "Sequence[TPSA] | object") -> "TPSA":
        """Compose this TPSA with a substitution map."""

        from .map import TPSAMap

        sub_map = substitutions if isinstance(substitutions, TPSAMap) else TPSAMap(substitutions)  # type: ignore[arg-type]
        return TPSAMap([self]).compose(sub_map)[0]

    def norm(self) -> float:
        self._assert_open()
        return float(load_library().mad_tpsa_nrm(self._ptr))

    def almost_equal(self, other: "TPSA | Number", *, tol: float = 0.0) -> bool:
        other = self._coerce_tpsa(other)
        return bool(load_library().mad_tpsa_equ(self._ptr, other._ptr, float(tol)))

    def is_null(self) -> bool:
        self._assert_open()
        return bool(load_library().mad_tpsa_isnul(self._ptr))

    def is_scalar(self) -> bool:
        self._assert_open()
        return bool(load_library().mad_tpsa_isval(self._ptr))

    def is_valid(self) -> bool:
        self._assert_open()
        return bool(load_library().mad_tpsa_isvalid(self._ptr))

    def _new_like(self) -> "TPSA":
        self._assert_open()
        ptr = load_library().mad_tpsa_new(self._ptr, same_order())
        return TPSA._from_ptr(self._desc, ptr)

    def _unary(self, c_function_name: str) -> "TPSA":
        out = self._new_like()
        getattr(load_library(), c_function_name)(self._ptr, out._ptr)
        return out

    def _binary(self, other: "TPSA | Number", c_function_name: str) -> "TPSA":
        other = self._coerce_tpsa(other)
        out = self._new_like()
        getattr(load_library(), c_function_name)(self._ptr, other._ptr, out._ptr)
        return out

    def _coerce_tpsa(self, other: "TPSA | Number") -> "TPSA":
        if isinstance(other, TPSA):
            other._assert_open()
            self._assert_compatible(other)
            return other
        if _is_number(other):
            return TPSA.constant(self.descriptor, float(other), order=self.order)
        return NotImplemented  # type: ignore[return-value]

    def _assert_compatible(self, other: "TPSA") -> None:
        if self.descriptor.address != other.descriptor.address:
            raise ValueError("TPSA descriptors differ")

    def _assert_open(self) -> None:
        if self._closed or self._ptr == ffi.NULL:
            raise TPSAClosedError("TPSA value has been closed")
        self._desc._assert_open()

    def close(self) -> None:
        if not self._closed and self._owns:
            load_library().mad_tpsa_del(self._ptr)
        self._closed = True
        self._ptr = ffi.NULL

    def __getitem__(self, monomial: Sequence[int] | int | str | tuple[()]) -> float:
        self._assert_open()
        lib = load_library()
        if monomial == ():
            return float(lib.mad_tpsa_geti(self._ptr, 0))
        if isinstance(monomial, int):
            if monomial < 0:
                raise IndexError("coefficient index must be non-negative")
            return float(lib.mad_tpsa_geti(self._ptr, int(monomial)))
        if isinstance(monomial, str):
            # MAD-NG strings represent dense monomial orders, e.g. "201".
            values = tuple(int(ch, 36) for ch in monomial.strip())
        else:
            values = tuple(int(v) for v in monomial)
        m = _monomial_array(values, self.descriptor.n_total)
        return float(lib.mad_tpsa_getm(self._ptr, self.descriptor.n_total, m))

    def __setitem__(self, monomial: Sequence[int] | int | str | tuple[()], value: float) -> None:
        self._assert_open()
        lib = load_library()
        if monomial == ():
            lib.mad_tpsa_seti(self._ptr, 0, 0.0, float(value))
            return
        if isinstance(monomial, int):
            if monomial < 0:
                raise IndexError("coefficient index must be non-negative")
            lib.mad_tpsa_seti(self._ptr, int(monomial), 0.0, float(value))
            return
        if isinstance(monomial, str):
            values = tuple(int(ch, 36) for ch in monomial.strip())
        else:
            values = tuple(int(v) for v in monomial)
        m = _monomial_array(values, self.descriptor.n_total)
        lib.mad_tpsa_setm(self._ptr, self.descriptor.n_total, m, 0.0, float(value))

    def __add__(self, other: "TPSA | Number"):
        if _is_number(other):
            out = self.copy()
            load_library().mad_tpsa_seti(out._ptr, 0, 1.0, float(other))
            return out
        if isinstance(other, TPSA):
            return self._binary(other, "mad_tpsa_add")
        return NotImplemented

    def __radd__(self, other: Number):
        return self.__add__(other)

    def __sub__(self, other: "TPSA | Number"):
        if _is_number(other):
            out = self.copy()
            load_library().mad_tpsa_seti(out._ptr, 0, 1.0, -float(other))
            return out
        if isinstance(other, TPSA):
            return self._binary(other, "mad_tpsa_sub")
        return NotImplemented

    def __rsub__(self, other: Number):
        if _is_number(other):
            out = self._new_like()
            load_library().mad_tpsa_scl(self._ptr, -1.0, out._ptr)
            load_library().mad_tpsa_seti(out._ptr, 0, 1.0, float(other))
            return out
        return NotImplemented

    def __mul__(self, other: "TPSA | Number"):
        if _is_number(other):
            out = self._new_like()
            load_library().mad_tpsa_scl(self._ptr, float(other), out._ptr)
            return out
        if isinstance(other, TPSA):
            return self._binary(other, "mad_tpsa_mul")
        return NotImplemented

    def __rmul__(self, other: Number):
        return self.__mul__(other)

    def __truediv__(self, other: "TPSA | Number"):
        if _is_number(other):
            out = self._new_like()
            load_library().mad_tpsa_divn(self._ptr, float(other), out._ptr)
            return out
        if isinstance(other, TPSA):
            return self._binary(other, "mad_tpsa_div")
        return NotImplemented

    def __rtruediv__(self, other: Number):
        if _is_number(other):
            out = self._new_like()
            load_library().mad_tpsa_inv(self._ptr, float(other), out._ptr)
            return out
        return NotImplemented

    def __pow__(self, other: "TPSA | Number"):
        self._assert_open()
        out = self._new_like()
        lib = load_library()
        if isinstance(other, bool):
            other = int(other)
        if isinstance(other, int):
            lib.mad_tpsa_powi(self._ptr, int(other), out._ptr)
            return out
        if isinstance(other, Real):
            lib.mad_tpsa_pown(self._ptr, float(other), out._ptr)
            return out
        if isinstance(other, TPSA):
            self._assert_compatible(other)
            lib.mad_tpsa_pow(self._ptr, other._ptr, out._ptr)
            return out
        return NotImplemented

    def __rpow__(self, other: Number):
        if _is_number(other):
            if float(other) <= 0:
                raise ValueError("scalar ** TPSA is implemented as exp(log(scalar) * TPSA), so scalar must be positive")
            from .functions import exp

            return exp(_math.log(float(other)) * self)
        return NotImplemented

    def __neg__(self) -> "TPSA":
        out = self._new_like()
        load_library().mad_tpsa_scl(self._ptr, -1.0, out._ptr)
        return out

    def __pos__(self) -> "TPSA":
        return self.copy()

    def __abs__(self) -> "TPSA":
        return self._unary("mad_tpsa_abs")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TPSA) or _is_number(other):
            try:
                return self.almost_equal(other)  # type: ignore[arg-type]
            except Exception:
                return False
        return False

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __float__(self) -> float:
        return self.constant_term

    def __iter__(self):
        return self.coefficients()

    def __enter__(self) -> "TPSA":
        self._assert_open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - destructor timing is interpreter-specific
        try:
            if getattr(self, "_owns", False) and not getattr(self, "_closed", True):
                lib = load_library()
                lib.mad_tpsa_del(self._ptr)
                self._closed = True
        except Exception:
            pass

    def __repr__(self) -> str:
        if self._closed:
            return "TPSA(closed=True)"
        terms = list(self.coefficients())
        shown = ", ".join(f"{m}: {v:.6g}" for m, v in terms[:6])
        if len(terms) > 6:
            shown += ", ..."
        return f"TPSA({{{shown}}}, order={self.order}, descriptor=0x{self.descriptor.address:x})"


def _is_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def coerce_common(a: TPSA | Number, b: TPSA | Number) -> tuple[TPSA, TPSA]:
    """Coerce two operands to TPSA values using the descriptor of either one."""

    if isinstance(a, TPSA) and isinstance(b, TPSA):
        a._assert_compatible(b)
        return a, b
    if isinstance(a, TPSA) and _is_number(b):
        return a, TPSA.constant(a.descriptor, float(b), order=a.order)
    if isinstance(b, TPSA) and _is_number(a):
        return TPSA.constant(b.descriptor, float(a), order=b.order), b
    raise TypeError("at least one operand must be a TPSA value")


def from_mapping(descriptor: Descriptor, terms: Mapping[Sequence[int], float], *, order: int | None = None) -> TPSA:
    """Create a TPSA from a mapping of monomial tuples to coefficients."""

    out = TPSA.constant(descriptor, 0.0, order=order)
    for monomial, value in terms.items():
        out[monomial] = value
    return out
        """Create a real constant TPSA value."""

        """Create a one-based real TPSA variable."""

        """Create a one-based real TPSA parameter."""

        """Return the descriptor used by this TPSA value."""

        """Return the underlying tpsa_t CFFI pointer."""

        """Return the integer address of the wrapped tpsa_t pointer."""

        """Return the allocated maximum Taylor order."""

        """Return the highest currently non-zero Taylor order."""

        """Return the allocated coefficient-vector length."""

        """Return the scalar coefficient of the zero monomial."""

        """Clear all coefficients in place and return self."""

        """Refresh internal low/high order metadata after coefficient changes."""

        """Set one coefficient and return self."""

        """Return all non-zero coefficients as a dictionary."""

        """Return the homogeneous component of the requested order."""

        """Return a copy with high or low orders removed according to MAD-NG semantics."""

        """Clear one homogeneous order in place and return self."""

        """Return the derivative with respect to a one-based variable index."""

        """Return the integral with respect to a one-based variable index."""

        """Return the Poisson bracket with another TPSA value."""

        """Return the MAD-NG TPSA norm."""

        """Compare two TPSA values using a coefficient tolerance."""

        """Return True when this TPSA is identically zero."""

        """Return True when this TPSA has no non-constant terms."""

        """Return True when the wrapped C TPSA object passes MAD-NG validation."""

        """Release the wrapped tpsa_t pointer if owned by this object."""

        """Return a coefficient by monomial tuple, string, or raw index."""

        """Set a coefficient by monomial tuple, string, or raw index."""

        """Return the TPSA sum with another TPSA-compatible value."""

        """Return the reflected TPSA sum."""

        """Return the TPSA difference with another TPSA-compatible value."""

        """Return the reflected TPSA difference."""

        """Return the TPSA product with another TPSA-compatible value."""

        """Return the reflected TPSA product."""

        """Return the TPSA quotient by another TPSA-compatible value."""

        """Return the reflected TPSA quotient."""

        """Raise this TPSA to an integer, real, or TPSA power."""

        """Raise a real scalar to this TPSA power."""

        """Return the additive inverse of this TPSA."""

        """Return a copy of this TPSA."""

        """Return the TPSA absolute value."""

        """Return coefficient-wise equality with a compatible value."""

        """Return coefficient-wise inequality with a compatible value."""

        """Iterate over non-zero coefficient pairs."""

        """Return this TPSA for use as a context manager."""

        """Close this TPSA when leaving a context-manager block."""

        """Return a concise representation with a few non-zero coefficients."""

