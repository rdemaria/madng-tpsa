# madng-tpsa

`madng-tpsa` is a self-contained Python package exposing MAD-NG-compatible real and complex TPSA APIs through CFFI. It provides:

- descriptor construction via `DescriptorBuilder` and `descriptor(...)`;
- scalar TPSA values via `TPSA`;
- TPSA maps/vectors via `TPSAMap` and `CTPSAMap`;
- mathematical functions such as `sin`, `cos`, `exp`, `log`, `sqrt`, `atan2`, `hypot`, `conj`, `real`, and `imag`;
- Python operator overloading so TPSA algebra looks like ordinary Python algebra.

The import package is `madng_tpsa`:

```python
import madng_tpsa as mt
```

## Status

This version builds a bundled CFFI extension named `madng_tpsa._madng_tpsa_cffi` at install time. By default, descriptors and TPSA values use that private C backend, so neither MAD-NG nor `pymadng` is a runtime dependency.

The bundled backend exports the MAD-NG real and complex TPSA C symbols used by the Python layer, including the map inverse entry points `mad_tpsa_minv` and `mad_ctpsa_minv`. Map inversion is LAPACK-backed: the C backend calls LAPACK `dgesv_` to invert the linear part and then performs the TPSA Newton inverse in C. The wrapper can still target an external, optimized MAD-NG/libgtpsa shared library with the same public symbols. MAD-NG and this package are GPLv3-or-later.

At runtime:

```python
import madng_tpsa as mt

print(mt.is_available())          # True when the bundled extension built correctly
print(mt.loaded_library_path())   # usually "vendored:madng_tpsa._madng_tpsa_cffi"
```

## Install

From a checkout of this repository:

```bash
python -m pip install .
python -m pip install -e '.[test]'
```

Building from source needs a C compiler, `cffi`, and LAPACK/BLAS development libraries. Wheels built from this tree include the compiled extension for the target platform, but source builds must be able to link `-llapack -lblas` by default.

Typical system packages:

```bash
# Debian/Ubuntu
sudo apt-get install build-essential liblapack-dev libblas-dev

# Fedora
sudo dnf install gcc lapack-devel blas-devel

# macOS with Homebrew
brew install lapack openblas
```

If your platform exposes LAPACK through another library name, set `MADNG_TPSA_LAPACK_LIBRARIES` before building. For example:

```bash
MADNG_TPSA_LAPACK_LIBRARIES=openblas python -m pip install .
MADNG_TPSA_LAPACK_LIBRARIES=lapack,blas python -m pip install .
```

## Optional: use an external MAD-NG/libgtpsa shared library

The loader still supports external libraries exporting MAD-NG TPSA symbols such as `mad_desc_newv`, `mad_tpsa_newd`, `mad_tpsa_add`, `mad_tpsa_compose`, `mad_ctpsa_newd`, and `mad_ctpsa_compose`:

```bash
export MADNG_TPSA_LIBRARY=/absolute/path/to/libmadng_tpsa.so
python - <<'PY'
import madng_tpsa as mt
print(mt.loaded_library_path())
PY
```

If an explicit external path fails to load, the package falls back to its bundled C core and keeps the external-load diagnostic in `mt.availability_error()`. The bundled backend is the normal path; the external path exists for developers who want to compare with, or benchmark against, an upstream MAD-NG/libgtpsa build.

To build an upstream MAD-NG/libgtpsa shared object for that override, use:

```bash
git clone https://github.com/MethodicalAcceleratorDesign/MAD-NG.git
cd MAD-NG/src
make -f Makefile.linux libmad.a

# From the madng-tpsa checkout:
python tools/build_madng_tpsa_shared.py /path/to/MAD-NG/src -o /path/to/libmadng_tpsa.so
export MADNG_TPSA_LIBRARY=/path/to/libmadng_tpsa.so
```

## Basic usage

```python
import madng_tpsa as mt

# Two variables, Taylor order 5.
desc = mt.DescriptorBuilder().variables(2).order(5).build()
x, y = desc.variables()

f = mt.sin(x) + y**2 + 3.0 * x * y + 2.0

print(f[0, 0])     # constant term: 2
print(f[1, 0])     # x coefficient: 1
print(f[3, 0])     # x^3 coefficient from sin(x): -1/6
print(f[0, 2])     # y^2 coefficient: 1
print(f.evaluate([0.2, 0.1]))
```

Dense monomial tuples use the descriptor's full dimension count: `n_variables + n_parameters`. For a two-variable descriptor, `f[3, 0]` means the coefficient of `x^3 y^0`.

## Descriptor builder

```python
desc = (
    mt.DescriptorBuilder()
    .variables(6)
    .order(5)
    .parameters(2, order=2)
    .build()
)
```

Shortcut:

```python
desc = mt.descriptor(6, 5, n_parameters=2, parameter_order=2)
```

Per-dimension order limits may be supplied with `orders(...)`:

```python
desc = (
    mt.DescriptorBuilder()
    .variables(2)
    .order(5)
    .parameters(1, order=2)
    .orders(5, 5, 2)
    .build()
)
```

## TPSA values

Construct constants, variables, and parameters:

```python
desc = mt.descriptor(2, 4)
x = mt.TPSA.variable(desc, 1)
y = mt.TPSA.variable(desc, 2, value=0.0, scale=1.0)
c = mt.TPSA.constant(desc, 3.5)
```

Operators are overloaded:

```python
f = (x + 2*y)**3 / (1 + x) - mt.log(1 + y)
g = mt.sqrt(1 + x**2 + y**2)
h = mt.atan2(y, x + 1)
```

Useful methods:

```python
dfdx = f.derivative(1)       # one-based variable index
int_y = f.integrate(2)
only_order_3 = f.get_order(3)
truncated = f.cut_order(4)
print(f.to_dict())           # {(monomial_tuple): coefficient, ...}
```

## Complex TPSA values

The package also wraps the complex MAD-NG CTPSA API (`mad_ctpsa_*`) through `CTPSA` and `CTPSAMap`. Python passes complex scalar arguments through real/imaginary `_r` entry points, avoiding CFFI ABI ambiguity for complex-by-value arguments while still using the C CTPSA backend for algebra, functions, composition, evaluation, and map inversion.

```python
import madng_tpsa as mt

desc = mt.descriptor(2, 5)
x, y = desc.variables()

z = mt.CTPSA.from_tpsa(x) + 1j * mt.CTPSA.from_tpsa(y)
f = mt.exp(1j * z) + z**2

print(f[1, 0])       # coefficient of x
print(f.real[2, 0])  # real projection as a TPSA
print(f.imag[1, 1])  # imaginary projection as a TPSA

cm = mt.CTPSAMap([z, mt.conj(z)])
print(cm.evaluate([0.1, 0.2]))
```

Complex maps mirror the real map API:

```python
identity = mt.CTPSAMap.identity(desc)
composed = cm @ identity
translated = cm.translate([0.01 + 0.0j, 0.0])
```

## TPSA maps

A `TPSAMap` is a vector of TPSA components. Arithmetic is component-wise; `@` means composition.

```python
desc = mt.descriptor(2, 5)
x, px = desc.variables()

identity = mt.TPSAMap.identity(desc)
drift = mt.TPSAMap([x + 1.2 * px, px])
kick = mt.TPSAMap([x, px - 0.05 * x**2])

one_turn = drift @ kick @ drift
print(one_turn.evaluate([1e-3, 0.0]))

inverse = one_turn.inverse()
print((inverse @ one_turn).evaluate([1e-3, 0.0]))
```

`TPSAMap.inverse()` calls the C `mad_tpsa_minv` entry point. In the bundled backend that routine is linked to LAPACK/BLAS; if you set `MADNG_TPSA_LIBRARY`, the same Python call dispatches to the external library's `mad_tpsa_minv`.

## Mathematical functions

The top-level namespace mirrors the TPSA C functions where practical:

```python
mt.sqrt(x)
mt.exp(x)
mt.log(x)
mt.sin(x)
mt.cos(x)
mt.tan(x)
mt.sinh(x)
mt.cosh(x)
mt.atan2(y, x)
mt.hypot(x, y)
mt.erf(x)
mt.erfc(x)
```

The same functions work element-wise on `TPSAMap` values.

## Tests

```bash
pytest
```

The test suite runs against the bundled LAPACK-linked C core by default. In the build environment used for this artifact, the suite reports `11 passed`.

## Examples

```bash
python examples/basic_polynomial.py
python examples/maps.py
python examples/parameters.py
python examples/parametric_normal_form.py
python examples/complex_tpsa.py
```

## Notes and limitations

- The wrapper targets both the real TPSA API (`mad_tpsa_*`) and a useful subset of the complex TPSA API (`mad_ctpsa_*`).
- The bundled backend is designed for portability and correctness on small/medium descriptors. For MAD-NG's full optimized performance profile, build or point to an external MAD-NG/libgtpsa shared library.
- Source builds require LAPACK/BLAS because map inversion is part of the supported TPSA algebra.
- Descriptor destruction is delicate in C APIs: no TPSA may still use a descriptor when `mad_desc_del` is called. Prefer normal Python object lifetime over explicit `Descriptor.close()` unless you know the TPSA values are gone.
## Public API summary

The package exports the following user-facing Python API. Each entry below is also documented with a Python docstring.

### Top-level helpers

- `descriptor(...)` — Shortcut for `Descriptor.create(...)`.
- `from_mapping(...)` — Create a TPSA from a mapping of monomial tuples to coefficients.
- `complex_from_mapping(...)` — Create a complex TPSA from a mapping of monomial tuples to coefficients.
- `load_library(...)` — Load and return the TPSA C library.
- `is_available(...)` — Return True when a MAD-NG-compatible TPSA backend can be loaded.
- `availability_error(...)` — Return the last backend-loading diagnostic message, if any.
- `loaded_library_path(...)` — Return the external path or vendored identifier for the loaded library.

### Descriptors

#### `Descriptor`

- `create(...)` — Create a descriptor.
- `n_total(...)` — Total monomial dimensions: variables plus parameters.
- `ptr(...)` — The underlying ``const desc_t *`` CFFI pointer.
- `address(...)` — Integer address of the underlying descriptor pointer.
- `maxlen(...)` — Return MAD-NG's maximum coefficient length at ``order``.
- `is_valid_monomial(...)` — Return whether a dense monomial tuple is valid for this descriptor.
- `monomial_index(...)` — Return MAD-NG's coefficient index for a dense monomial tuple.
- `monomial(...)` — Return the dense monomial tuple for a coefficient index.
- `constant(...)` — Create a scalar TPSA value using this descriptor.
- `variable(...)` — Create the one-based MAD-NG variable ``index`` as a TPSA value.
- `variables(...)` — Return all descriptor variables as TPSA values.
- `parameter(...)` — Create the one-based MAD-NG parameter ``index`` as a TPSA value.
- `complex_constant(...)` — Create a complex scalar TPSA value using this descriptor.
- `complex_variable(...)` — Create the one-based MAD-NG variable ``index`` as a complex TPSA value.
- `complex_variables(...)` — Return all descriptor variables as complex TPSA values.
- `complex_parameter(...)` — Create the one-based MAD-NG parameter ``index`` as a complex TPSA value.
- `close(...)` — Release this descriptor.

#### `DescriptorBuilder`

- `variables(...)` — Set the number of real TPSA variables for the descriptor being built.
- `order(...)` — Set the maximum Taylor order for the descriptor being built.
- `parameters(...)` — Set the number and optional order of TPSA parameters.
- `orders(...)` — Set per-variable and per-parameter maximum orders.
- `build(...)` — Build and return a Descriptor from the configured options.

### Real TPSA values and maps

#### `TPSA`

- `constant(...)` — Create a real constant TPSA value.
- `variable(...)` — Create a one-based real TPSA variable.
- `parameter(...)` — Create a one-based real TPSA parameter.
- `descriptor(...)` — Return the descriptor used by this TPSA value.
- `ptr(...)` — Return the underlying tpsa_t CFFI pointer.
- `address(...)` — Return the integer address of the wrapped tpsa_t pointer.
- `order(...)` — Return the allocated maximum Taylor order.
- `high_order(...)` — Return the highest currently non-zero Taylor order.
- `length(...)` — Return the allocated coefficient-vector length.
- `constant_term(...)` — Return the scalar coefficient of the zero monomial.
- `copy(...)` — Return a copy of this TPSA value.
- `clear(...)` — Clear all coefficients in place and return self.
- `update(...)` — Refresh internal low/high order metadata after coefficient changes.
- `coefficient(...)` — Return a coefficient.
- `set_coefficient(...)` — Set one coefficient and return self.
- `coefficients(...)` — Yield non-zero coefficients as ``(monomial, value)`` pairs.
- `to_dict(...)` — Return all non-zero coefficients as a dictionary.
- `get_order(...)` — Return the homogeneous component of the requested order.
- `cut_order(...)` — Return a copy with high or low orders removed according to MAD-NG semantics.
- `clear_order(...)` — Clear one homogeneous order in place and return self.
- `derivative(...)` — Return the derivative with respect to a one-based variable index.
- `integrate(...)` — Return the integral with respect to a one-based variable index.
- `poisson_bracket(...)` — Return the Poisson bracket with another TPSA value.
- `evaluate(...)` — Evaluate this TPSA for a vector of variable/parameter values.
- `compose(...)` — Compose this TPSA with a substitution map.
- `norm(...)` — Return the MAD-NG TPSA norm.
- `almost_equal(...)` — Compare two TPSA values using a coefficient tolerance.
- `is_null(...)` — Return True when this TPSA is identically zero.
- `is_scalar(...)` — Return True when this TPSA has no non-constant terms.
- `is_valid(...)` — Return True when the wrapped C TPSA object passes MAD-NG validation.
- `close(...)` — Release the wrapped tpsa_t pointer if owned by this object.
- operator overloads — Supports `+`, `-`, `*`, `/`, `**`, unary signs, coefficient indexing, iteration, equality, and context-manager cleanup.

#### `TPSAMap`

- `identity(...)` — Return an identity map ``[x1, x2, ...]``.
- `constants(...)` — Create a TPSA map whose components are constants.
- `descriptor(...)` — Return the descriptor shared by all map components.
- `size(...)` — Return the number of components in the map.
- `copy(...)` — Return a component-wise copy of this map.
- `apply(...)` — Apply a function to each component and return a new map.
- `compose(...)` — Compose this map with ``inner`` using ``mad_tpsa_compose``.
- `inverse(...)` — Return the formal inverse map using ``mad_tpsa_minv``.
- `translate(...)` — Translate/evaluate this map around a numeric vector.
- `evaluate(...)` — Numerically evaluate this TPSA map.
- `norm(...)` — Return the MAD-NG norm of the map.
- `max_order(...)` — Return the maximum order among the map components.
- operator overloads — Supports component-wise arithmetic, `@` composition, indexing, iteration, and copying.

### Complex TPSA values and maps

#### `CTPSA`

- `constant(...)` — Create a complex constant TPSA value.
- `variable(...)` — Create a one-based complex TPSA variable.
- `parameter(...)` — Create a one-based complex TPSA parameter.
- `from_tpsa(...)` — Promote real and optional imaginary TPSA values to one complex TPSA.
- `descriptor(...)` — Return the descriptor used by this complex TPSA value.
- `ptr(...)` — Return the underlying ctpsa_t CFFI pointer.
- `address(...)` — Return the integer address of the wrapped ctpsa_t pointer.
- `order(...)` — Return the allocated maximum Taylor order.
- `high_order(...)` — Return the highest currently non-zero Taylor order.
- `length(...)` — Return the allocated coefficient-vector length.
- `constant_term(...)` — Return the complex coefficient of the zero monomial.
- `real(...)` — Return the real projection as a TPSA value.
- `imag(...)` — Return the imaginary projection as a TPSA value.
- `abs(...)` — Return the complex magnitude as a real TPSA value.
- `arg(...)` — Return the complex phase as a real TPSA value.
- `copy(...)` — Return a copy of this complex TPSA value.
- `clear(...)` — Clear all coefficients in place and return self.
- `update(...)` — Refresh internal low/high order metadata after coefficient changes.
- `coefficient(...)` — Return a complex coefficient.
- `set_coefficient(...)` — Set one complex coefficient and return self.
- `coefficients(...)` — Yield non-zero complex coefficients as monomial-value pairs.
- `to_dict(...)` — Return all non-zero complex coefficients as a dictionary.
- `get_order(...)` — Return the homogeneous component of the requested order.
- `cut_order(...)` — Return a copy with high or low orders removed according to MAD-NG semantics.
- `clear_order(...)` — Clear one homogeneous order in place and return self.
- `derivative(...)` — Return the derivative with respect to a one-based variable index.
- `integrate(...)` — Return the integral with respect to a one-based variable index.
- `poisson_bracket(...)` — Return the complex Poisson bracket with another value.
- `evaluate(...)` — Evaluate this complex TPSA for a vector of values.
- `compose(...)` — Compose this complex TPSA with a substitution map.
- `norm(...)` — Return the MAD-NG complex TPSA norm.
- `almost_equal(...)` — Compare two complex TPSA values using a coefficient tolerance.
- `is_null(...)` — Return True when this complex TPSA is identically zero.
- `is_scalar(...)` — Return True when this complex TPSA has no non-constant terms.
- `is_valid(...)` — Return True when the wrapped C CTPSA object passes validation.
- `close(...)` — Release the wrapped ctpsa_t pointer if owned by this object.
- `conjugate(...)` — Return the complex conjugate of this TPSA.
- operator overloads — Supports complex `+`, `-`, `*`, `/`, `**`, unary signs, coefficient indexing, iteration, equality, conversion to `complex`, and context-manager cleanup.

#### `CTPSAMap`

- `identity(...)` — Return a complex identity map.
- `descriptor(...)` — Return the descriptor shared by all complex map components.
- `copy(...)` — Return a component-wise copy of this complex map.
- `apply(...)` — Apply a function to each component and return a new complex map.
- `compose(...)` — Compose this complex map with an inner map.
- `translate(...)` — Translate/evaluate this complex map around a numeric vector.
- `evaluate(...)` — Numerically evaluate this complex TPSA map.
- `inverse(...)` — Return the formal inverse complex map using MAD-NG CTPSA inversion.
- `norm(...)` — Return the MAD-NG norm of the complex map.
- operator overloads — Supports component-wise complex arithmetic, `@` composition, indexing, iteration, and copying.

### Mathematical functions

- `unit(...)` — Return the normalized sign/unit TPSA function.
- `abs(...)` — Return the absolute value for real TPSA or magnitude for complex TPSA.
- `conj(...)` — Return the complex conjugate of a CTPSA or CTPSAMap.
- `real(...)` — Return the real projection of a supported TPSA value.
- `imag(...)` — Return the imaginary projection of a supported TPSA value.
- `carg(...)` — Return the complex phase of a CTPSA value.
- `sqrt(...)` — Return the square root of a scalar, TPSA, CTPSA, or map.
- `exp(...)` — Return the exponential of a scalar, TPSA, CTPSA, or map.
- `log(...)` — Return the natural logarithm of a scalar, TPSA, CTPSA, or map.
- `sin(...)` — Return the sine of a scalar, TPSA, CTPSA, or map.
- `cos(...)` — Return the cosine of a scalar, TPSA, CTPSA, or map.
- `tan(...)` — Return the tangent of a scalar, TPSA, CTPSA, or map.
- `cot(...)` — Return the cotangent of a scalar, TPSA, CTPSA, or map.
- `sinc(...)` — Return sin(x)/x for a scalar, TPSA, CTPSA, or map.
- `sinh(...)` — Return the hyperbolic sine of a scalar, TPSA, CTPSA, or map.
- `cosh(...)` — Return the hyperbolic cosine of a scalar, TPSA, CTPSA, or map.
- `tanh(...)` — Return the hyperbolic tangent of a scalar, TPSA, CTPSA, or map.
- `coth(...)` — Return the hyperbolic cotangent of a scalar, TPSA, CTPSA, or map.
- `sinhc(...)` — Return sinh(x)/x for a scalar, TPSA, CTPSA, or map.
- `asin(...)` — Return the inverse sine of a scalar, TPSA, CTPSA, or map.
- `acos(...)` — Return the inverse cosine of a scalar, TPSA, CTPSA, or map.
- `atan(...)` — Return the inverse tangent of a scalar, TPSA, CTPSA, or map.
- `atan2(...)` — Return real atan2 for TPSA-compatible real arguments.
- `acot(...)` — Return the inverse cotangent of a scalar, TPSA, CTPSA, or map.
- `asinc(...)` — Return the inverse-sinc MAD-NG TPSA function where supported.
- `asinh(...)` — Return the inverse hyperbolic sine of a scalar, TPSA, CTPSA, or map.
- `acosh(...)` — Return the inverse hyperbolic cosine of a scalar, TPSA, CTPSA, or map.
- `atanh(...)` — Return the inverse hyperbolic tangent of a scalar, TPSA, CTPSA, or map.
- `acoth(...)` — Return the inverse hyperbolic cotangent of a scalar, TPSA, CTPSA, or map.
- `asinhc(...)` — Return asinh(x)/x through the MAD-NG TPSA backend where supported.
- `erf(...)` — Return the error function of a scalar, TPSA, CTPSA, or map.
- `erfc(...)` — Return the complementary error function of a scalar, TPSA, CTPSA, or map.
- `erfcx(...)` — Return the scaled complementary error function.
- `erfi(...)` — Return the imaginary error function through the MAD-NG backend.
- `wf(...)` — Return the Faddeeva-related MAD-NG special function.
- `invsqrt(...)` — Return scale divided by the square root of a supported value.
- `sincos(...)` — Return sine and cosine together.
- `sincosh(...)` — Return hyperbolic sine and cosine together.
- `hypot(...)` — Return sqrt(x*x + y*y) for supported values.
- `hypot3(...)` — Return sqrt(x*x + y*y + z*z) for supported values.
