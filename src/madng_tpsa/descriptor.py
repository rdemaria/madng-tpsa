"""Descriptor construction for MAD-NG TPSA values."""

from __future__ import annotations

from typing import Iterable, Sequence

from ._cffi import ffi, load_library, _ord, ptr_address
from .exceptions import DescriptorClosedError, MadngTPSAError


class Descriptor:
    """A MAD-NG TPSA descriptor.

    A descriptor defines the number of variables, the maximum Taylor order, and
    optionally the number/order of parameters. TPSA values keep a reference to
    their descriptor, so a descriptor created by this class stays alive as long
    as any Python TPSA object uses it.
    """

    __slots__ = (
        "_ptr",
        "_owns",
        "_closed",
        "n_variables",
        "max_order",
        "n_parameters",
        "parameter_order",
    )

    def __init__(
        self,
        ptr,
        *,
        owns: bool = True,
        n_variables: int | None = None,
        max_order: int | None = None,
        n_parameters: int | None = None,
        parameter_order: int | None = None,
    ) -> None:
        if ptr == ffi.NULL:
            raise MadngTPSAError("MAD-NG returned a NULL descriptor")
        self._ptr = ptr
        self._owns = owns
        self._closed = False
        if None in (n_variables, max_order, n_parameters, parameter_order):
            n_variables, max_order, n_parameters, parameter_order = self._introspect(ptr)
        self.n_variables = int(n_variables)
        self.max_order = int(max_order)
        self.n_parameters = int(n_parameters)
        self.parameter_order = int(parameter_order)

    @classmethod
    def create(
        cls,
        n_variables: int,
        max_order: int,
        *,
        n_parameters: int = 0,
        parameter_order: int | None = None,
        orders: Sequence[int] | None = None,
    ) -> "Descriptor":
        """Create a descriptor.

        Parameters are one-based in MAD-NG, but descriptor counts are ordinary
        Python integers. ``orders`` may be used to pass the per-variable/per-
        parameter order vector required by ``mad_desc_newvpo``.
        """

        n_variables = _positive_int(n_variables, "n_variables")
        max_order = _positive_int(max_order, "max_order")
        n_parameters = _nonnegative_int(n_parameters, "n_parameters")
        if parameter_order is None:
            parameter_order = max_order if n_parameters else 0
        parameter_order = _nonnegative_int(parameter_order, "parameter_order")

        lib = load_library()
        if orders is not None:
            expected = n_variables + n_parameters
            order_tuple = tuple(_ord(int(order)) for order in orders)
            if len(order_tuple) != expected:
                raise ValueError(f"orders must contain {expected} entries, got {len(order_tuple)}")
            no = ffi.new("ord_t[]", order_tuple)
            ptr = lib.mad_desc_newvpo(
                n_variables,
                _ord(max_order),
                n_parameters,
                _ord(parameter_order),
                no,
            )
        elif n_parameters:
            ptr = lib.mad_desc_newvp(
                n_variables,
                _ord(max_order),
                n_parameters,
                _ord(parameter_order),
            )
        else:
            ptr = lib.mad_desc_newv(n_variables, _ord(max_order))
        return cls(ptr, owns=True)

    @staticmethod
    def _introspect(ptr) -> tuple[int, int, int, int]:
        lib = load_library()
        mo = ffi.new("ord_t *")
        np = ffi.new("int *")
        po = ffi.new("ord_t *")
        nv = lib.mad_desc_getnv(ptr, mo, np, po)
        return int(nv), int(mo[0]), int(np[0]), int(po[0])

    @property
    def n_total(self) -> int:
        """Total monomial dimensions: variables plus parameters."""

        return self.n_variables + self.n_parameters

    @property
    def ptr(self):
        """The underlying ``const desc_t *`` CFFI pointer."""

        self._assert_open()
        return self._ptr

    @property
    def address(self) -> int:
        """Integer address of the underlying descriptor pointer."""

        self._assert_open()
        return ptr_address(self._ptr)

    def maxlen(self, order: int | None = None) -> int:
        """Return MAD-NG's maximum coefficient length at ``order``."""

        self._assert_open()
        if order is None:
            order = self.max_order
        return int(load_library().mad_desc_maxlen(self._ptr, _ord(int(order))))

    def is_valid_monomial(self, monomial: Sequence[int]) -> bool:
        """Return whether a dense monomial tuple is valid for this descriptor."""

        self._assert_open()
        m = _monomial_array(monomial, self.n_total)
        return bool(load_library().mad_desc_isvalidm(self._ptr, self.n_total, m))

    def monomial_index(self, monomial: Sequence[int]) -> int:
        """Return MAD-NG's coefficient index for a dense monomial tuple."""

        self._assert_open()
        m = _monomial_array(monomial, self.n_total)
        return int(load_library().mad_desc_idxm(self._ptr, self.n_total, m))

    def monomial(self, index: int) -> tuple[int, ...]:
        """Return the dense monomial tuple for a coefficient index."""

        self._assert_open()
        if index < 0:
            raise ValueError("index must be non-negative")
        m = ffi.new("ord_t[]", self.n_total)
        load_library().mad_desc_mono(self._ptr, int(index), self.n_total, m, ffi.NULL)
        return tuple(int(m[i]) for i in range(self.n_total))

    def constant(self, value: float = 0.0, *, order: int | None = None):
        """Create a scalar TPSA value using this descriptor."""

        from .tpsa import TPSA

        return TPSA.constant(self, value, order=order)

    def variable(
        self,
        index: int,
        *,
        value: float = 0.0,
        scale: float = 1.0,
        order: int | None = None,
    ):
        """Create the one-based MAD-NG variable ``index`` as a TPSA value."""

        from .tpsa import TPSA

        return TPSA.variable(self, index, value=value, scale=scale, order=order)

    def variables(self, *, order: int | None = None) -> tuple:
        """Return all descriptor variables as TPSA values."""

        return tuple(self.variable(i + 1, order=order) for i in range(self.n_variables))

    def parameter(self, index: int, *, value: float = 0.0, order: int | None = None):
        """Create the one-based MAD-NG parameter ``index`` as a TPSA value."""

        from .tpsa import TPSA

        return TPSA.parameter(self, index, value=value, order=order)

    def complex_constant(self, value: complex = 0.0, *, order: int | None = None):
        """Create a complex scalar TPSA value using this descriptor."""

        from .ctpsa import CTPSA

        return CTPSA.constant(self, value, order=order)

    def complex_variable(
        self,
        index: int,
        *,
        value: complex = 0.0,
        scale: complex = 1.0,
        order: int | None = None,
    ):
        """Create the one-based MAD-NG variable ``index`` as a complex TPSA value."""

        from .ctpsa import CTPSA

        return CTPSA.variable(self, index, value=value, scale=scale, order=order)

    def complex_variables(self, *, order: int | None = None) -> tuple:
        """Return all descriptor variables as complex TPSA values."""

        return tuple(self.complex_variable(i + 1, order=order) for i in range(self.n_variables))

    def complex_parameter(self, index: int, *, value: complex = 0.0, order: int | None = None):
        """Create the one-based MAD-NG parameter ``index`` as a complex TPSA value."""

        from .ctpsa import CTPSA

        return CTPSA.parameter(self, index, value=value, order=order)

    def close(self) -> None:
        """Release this descriptor.

        MAD-NG requires that no TPSA still uses a descriptor when the descriptor
        is destroyed. Letting Python garbage collection manage descriptors is
        usually safer than calling this manually.
        """

        if not self._closed and self._owns:
            load_library().mad_desc_del(self._ptr)
        self._closed = True
        self._ptr = ffi.NULL

    def _assert_open(self) -> None:
        if self._closed or self._ptr == ffi.NULL:
            raise DescriptorClosedError("descriptor has been closed")

    def __enter__(self) -> "Descriptor":
        self._assert_open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - destructor timing is interpreter-specific
        try:
            if getattr(self, "_owns", False) and not getattr(self, "_closed", True):
                lib = load_library()
                lib.mad_desc_del(self._ptr)
                self._closed = True
        except Exception:
            pass

    def __repr__(self) -> str:
        state = "closed" if self._closed else f"0x{self.address:x}"
        return (
            "Descriptor("
            f"n_variables={self.n_variables}, max_order={self.max_order}, "
            f"n_parameters={self.n_parameters}, parameter_order={self.parameter_order}, "
            f"ptr={state})"
        )


class DescriptorBuilder:
    """Fluent builder for :class:`Descriptor`.

    Examples
    --------
    >>> desc = DescriptorBuilder().variables(2).order(5).build()
    >>> x, px = desc.variables()
    """

    def __init__(
        self,
        n_variables: int | None = None,
        max_order: int | None = None,
        *,
        n_parameters: int = 0,
        parameter_order: int | None = None,
        orders: Iterable[int] | None = None,
    ) -> None:
        self.n_variables = n_variables
        self.max_order = max_order
        self.n_parameters = n_parameters
        self.parameter_order = parameter_order
        self.per_dimension_orders = None if orders is None else tuple(int(order) for order in orders)

    def variables(self, n_variables: int) -> "DescriptorBuilder":
        self.n_variables = _positive_int(n_variables, "n_variables")
        return self

    def order(self, max_order: int) -> "DescriptorBuilder":
        self.max_order = _positive_int(max_order, "max_order")
        return self

    def parameters(self, n_parameters: int, *, order: int | None = None) -> "DescriptorBuilder":
        self.n_parameters = _nonnegative_int(n_parameters, "n_parameters")
        self.parameter_order = None if order is None else _nonnegative_int(order, "parameter_order")
        return self

    def orders(self, *orders: int) -> "DescriptorBuilder":
        self.per_dimension_orders = tuple(_ord(int(order)) for order in orders)
        return self

    def build(self) -> Descriptor:
        if self.n_variables is None:
            raise ValueError("number of variables is required; call .variables(n)")
        if self.max_order is None:
            raise ValueError("maximum order is required; call .order(mo)")
        return Descriptor.create(
            self.n_variables,
            self.max_order,
            n_parameters=self.n_parameters,
            parameter_order=self.parameter_order,
            orders=self.per_dimension_orders,
        )

    def __repr__(self) -> str:
        return (
            "DescriptorBuilder("
            f"n_variables={self.n_variables}, max_order={self.max_order}, "
            f"n_parameters={self.n_parameters}, parameter_order={self.parameter_order}, "
            f"orders={self.per_dimension_orders})"
        )


def descriptor(
    n_variables: int,
    max_order: int,
    *,
    n_parameters: int = 0,
    parameter_order: int | None = None,
    orders: Sequence[int] | None = None,
) -> Descriptor:
    """Shortcut for :meth:`Descriptor.create`."""

    return Descriptor.create(
        n_variables,
        max_order,
        n_parameters=n_parameters,
        parameter_order=parameter_order,
        orders=orders,
    )


def _positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _nonnegative_int(value: int, name: str) -> int:
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def _monomial_array(monomial: Sequence[int], expected_len: int):
    values = tuple(int(x) for x in monomial)
    if len(values) != expected_len:
        raise ValueError(f"monomial must contain {expected_len} orders, got {len(values)}")
    return ffi.new("ord_t[]", [_ord(v) for v in values])
