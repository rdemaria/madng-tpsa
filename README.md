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
python examples/complex_tpsa.py
```

## Notes and limitations

- The wrapper targets both the real TPSA API (`mad_tpsa_*`) and a useful subset of the complex TPSA API (`mad_ctpsa_*`).
- The bundled backend is designed for portability and correctness on small/medium descriptors. For MAD-NG's full optimized performance profile, build or point to an external MAD-NG/libgtpsa shared library.
- Source builds require LAPACK/BLAS because map inversion is part of the supported TPSA algebra.
- Descriptor destruction is delicate in C APIs: no TPSA may still use a descriptor when `mad_desc_del` is called. Prefer normal Python object lifetime over explicit `Descriptor.close()` unless you know the TPSA values are gone.
