# Coupled 6D parametric normal form with an RF cavity

`parametric_normal_form_6d.py` extends the one-degree-of-freedom example to
three coupled canonical modes. It computes parameter-dependent closed orbits,
linear normalizing matrices, nonlinear canonical changes of coordinates, and
the full 3-by-3 leading amplitude-detuning matrix.

## Run

From the repository root, with a built LAPACK-linked extension:

```sh
python -m pip install -e '.[examples]'
python examples/parametric_normal_form_6d.py
python examples/parametric_normal_form_6d.py --parameter-order 2
python examples/parametric_normal_form_6d.py --parameter-order 2 --parameters ks vrf kxy
python -m pytest tests/test_normal_form_6d.py
```

NumPy is used only for small numerical eigenvalue/linear solves. The core
package does not acquire a mandatory NumPy dependency. Complex phase-space
algebra uses the package's native `CTPSA` C wrapper, not a real/imaginary pair
implementation. Parameter coefficients are calculated algebraically; finite
differences appear only in independent tests.

## Model and conventions

The phase coordinates are the dimensionless canonical pairs

```
u = (x, px, y, py, z, delta)
J = diag([[0, 1], [-1, 0]], [[0, 1], [-1, 0]], [[0, 1], [-1, 0]])
```

This is a scaled, illustrative symplectic ring, **not a calibrated machine or
an exact relativistic drift model**. In particular, `delta` is the canonical
momentum conjugate to `z` in this model, and `vrf` is a scaled kick strength,
not a voltage in volts.

The element order is half drift, QF with an octupole and skew quadrupole,
full drift, QD with a sextupole and horizontal dipole kick, half drift, RF
cavity. The drift Hamiltonian is

```text
T = (px^2 + py^2 + eta*delta^2)/2 + dispersion*px*delta
eta = 0.42, dispersion = 0.13
```

Consequently `x += L*(px + dispersion*delta)` **and**
`z += L*(eta*delta + dispersion*px)`. Both terms are needed to make the
horizontal-longitudinal coupling canonical. The skew quadrupole couples the
two transverse planes. The result is a genuinely coupled six-dimensional
linear map, rather than a 4D map with an independent longitudinal oscillator.

The thin multipole potentials and the RF kick are

```text
Vquad = k*(x^2-y^2)/2
Vskew = kxy*x*y
Vsext = ks*(x^3-3*x*y^2)/6
Voct  = ko*(x^4-6*x^2*y^2+y^4)/24
delta <- delta - vrf*sin(k_rf*z + phirf),  k_rf = 1.1
```

Momentum kicks are minus the corresponding coordinate derivatives of these
potentials. The dipole kick is `px += kb`. The full sine is used in numerical
tracking; TPSA tracking retains its phase Taylor terms through cubic order,
including the longitudinal amplitude detuning.

Default nominal settings are:

| Knob | Nominal value | Parameter increment |
|---|---:|---|
| `kf` | 0.86 | `dkf` |
| `kd` | -0.74 | `dkd` |
| `ks` | 0.06 | `dks` |
| `ko` | 0.025 | `dko` |
| `kxy` | 0.035 | `dkxy` |
| `vrf` | 0.18 | `dvrf` |
| `kb` | 0 | `dkb` |
| `phirf` | 0 | `dphirf`, in radians |

Increments are additive, not relative errors. Nonzero nominal `ks` makes the
sextupole-squared contribution visible even at first parameter order. At
parameter order two the output also contains explicit `dks^2`, mixed magnet/RF
terms, and quadratic/mixed sensitivities of the orbit and linear optics.
Nominal `kb` and `phirf` must be zero; their parameter increments are supported.
For example, the RF-phase dependence gives `z_co = -dphirf/k_rf` locally.

## Calculation

### 1. Parameter-dependent closed orbit

The code solves `M(c(p); p) = c(p)` in the truncated parameter ring. The
nominal fixed point is zero. A frozen inverse of `M'(0;0)-I` removes successive
parameter orders; the final residual is checked over **all** retained
parameter coefficients. Integer-tune/ill-conditioned fixed-point equations
are rejected. Tracking `c(p)+u` then subtracting `c(p)` produces the centered
map and its parameter-dependent Jacobian `R(p)`.

### 2. Linear symplectic normalizer

NumPy supplies nominal eigenvectors only. Each eigenpair is continued in the
parameter ring using the bordered linear system

```text
[ R0-lambda0*I   -v0 ] [ dv      ] = [ -(R*v-lambda*v) ]
[ l0              0 ] [ dlambda ]   [ -(l0*v-1)       ]
```

Here `l0` is a nominal left eigenvector, normalized by `l0*v0=1`. This fixes a
local phase/gauge rather than independently diagonalizing at sampled knob
values. Real and imaginary parts are normalized by their symplectic area.
The returned real `A(p)` satisfies

```text
A(p).T * J * A(p) = J
A(p)^-1 * R(p) * A(p) = diag(R(mu1), R(mu2), R(mu3))
R(mu) = [[cos(mu), sin(mu)], [-sin(mu), cos(mu)]]
```

Modes are labelled x-like, y-like, and synchrotron-like by nominal plane
participation. These are normal-mode labels, not independent physical x/y/z
oscillations. The labels and eigenvector gauge are continued locally in the
parameters. Unstable, degenerate, or ill-conditioned eigenpairs are rejected.

### 3. Nonlinear canonical term removal

With `z_j=(Q_j+i*P_j)/sqrt(2)`, `zbar_j=(Q_j-i*P_j)/sqrt(2)`, the formal variable
order is `(z1,zbar1,z2,zbar2,z3,zbar3)`. The Poisson tensor is `-i*J`. Physical
points have `zbar_j=conj(z_j)`, but the six formal variables are independent
when manipulating monomials. Native `CTPSA.conjugate()` conjugates coefficients;
it does not automatically exchange formal `z` and `zbar` variables.

Write the normalized map as `G(w)=Lambda*w+g2(w)+g3(w)`. For every
nonresonant monomial `w^m`, the homological equation is

```text
u_n[i,m] = g_n[i,m] / (product_j lambda_j^m_j - lambda_i)
```

All numerator and denominator coefficients remain parameter-dependent.
Structural resonances `z_i*J_k`, with `J_k=z_k*zbar_k`, are retained. A small
nonstructural denominator raises an error: this example does **not** pretend
to compute a valid nonresonant normal form on a coupling or synchro-betatron
resonance.

The quadratic removal is a canonical flow, not just the substitution `w+u2`.
At cubic order its half-flow correction must be retained:

```text
H2(w) = w + u2(w) + 1/2 * Du2(w)*u2(w)
g3_new = g3 + Dg2*u2 + 1/2*(Lambda*Du2*u2 - (Du2*u2)(Lambda*w))
H(w) = w + u2(w) + u3(w) + 1/2*Du2(w)*u2(w)
```

This is important for obtaining the sextupole-squared detuning correctly.
The script checks `G o H = H o N` coefficient by coefficient through cubic
phase order and checks the canonical brackets of `H` through phase degree two.
A cubic Taylor map determines its symplecticity defect only through degree two;
higher-degree bracket terms cannot be used as a test without higher map orders.

The full physical normalizing map exported as `nonlinear_A` is

```text
u = c(p) + A(p)*S*H(w;p)
S_pair = [[1,1],[-i,i]] / sqrt(2)
M o nonlinear_A = nonlinear_A o N + O(|w|^4)
```

Here `A` is the **linear real matrix**, while `H` is the nonlinear near-identity
map in complex normal coordinates. They are not interchangeable.

### 4. Parametric amplitude detuning

If the resonant normal map contains `C[j,k]*z_j*J_k`, then

```text
z'_j = exp(-i*2*pi*Q_j(p))*z_j + sum_k C[j,k](p)*z_j*J_k
D[j,k](p) = -imag(C[j,k](p)/lambda_j(p)) / (2*pi)
Q_j(J,p) = Q_j(0,p) + sum_k D[j,k](p)*J_k + higher action orders
```

The full 3-by-3 `D` includes transverse self/cross detuning, transverse-
longitudinal cross detuning, and synchrotron amplitude detuning. Its symmetry
`D[j,k]=D[k,j]` is checked for all retained parameter coefficients. Cubic map
order gives **linear-in-action detuning only**; it does not determine the
coefficients of `J_k*J_l` in the tune.

## Programmatic use and native exports

```python
import runpy

example = runpy.run_path('examples/parametric_normal_form_6d.py')
result = example['analyze'](parameter_order=2, parameters=('ks', 'vrf'))
native = result.native()

print(native['parameters'])           # ('ks', 'vrf')
print(native['A'][0][0])               # parameter-dependent native CTPSA
print(native['H'])                     # native CTPSAMap, near identity
print(native['nonlinear_A'])           # full complex-normal -> physical map
print(native['detuning'][0][2])        # D13(p), a native CTPSA

# Exponents: six phase variables, then the two parameters.
D11 = native['detuning'][0][0]
print(D11[(0,)*6 + (2,0)])             # coefficient of dks^2
print(D11[(0,)*6 + (1,1)])             # coefficient of dks*dvrf
```

A square Taylor coefficient is half the corresponding second derivative.
A mixed coefficient for two distinct parameters is the mixed second derivative.

The calculation stores parameter exponents outside the six-variable phase
`CTPSA` descriptor. This makes the retained orders rectangular: phase degree
at most three **and** parameter degree at most the chosen parameter order.
Export allocates a standard combined descriptor with total order
`3 + parameter_order` and explicit parameter limits, filling the retained
coefficients. It does not manufacture coefficients of higher phase degree.

**Native composition:** the current bundled backend treats missing
substitutions as zero. When composing exported maps, supply identity
substitutions for parameters as well as phase coordinates. The example's
`compose_native(outer, inner)` helper does this. Bare `outer @ inner` with only
six substitutions can otherwise discard parameter dependence. This example
does not change that existing backend behavior.

## Functions and outputs

| Entry point | Description |
|---|---|
| `analyze(...)` | Calculate the cubic normal form and its parameter Taylor coefficients. |
| `track(coordinates, knobs)` | Apply the symplectic ring using either numeric coordinates or parameter jets. |
| `closed_orbit(ring, knobs)` | Solve the parameter fixed-point equation and report its residual. |
| `linear_normalizer(matrix, ring)` | Continue eigenvectors and construct a real symplectic `A(p)`. |
| `homological(...)` | Divide nonresonant terms by their parameter-dependent homological denominators. |
| `symplectic_defect(H)` | Check all independent canonical brackets through the valid phase degree. |
| `Result.native()` | Export orbit, `A`, `H`, `N`, full normalizing map, tunes, and detuning as native objects. |
| `compose_native(outer, inner)` | Compose native maps while preserving explicit parameter identities. |
| `scalar_terms(value)` | Format a parameter-only result as a readable coefficient series. |
| `main(argv)` | Run the command-line demonstration and print its diagnostics. |

The `Jet` and `ParameterRing` classes are example-local bookkeeping helpers.
They are not new package-level public APIs.

## Tests and limitations

The tests include coefficient-level fixed-point/conjugacy residuals, coupled
linear symplecticity, nonlinear canonical brackets (also directly through the
C Poisson-bracket API), detuning symmetry, parameter derivatives checked against
independent numerical differences, square/mixed parameter coefficients, native
export, and the fourth-order conjugacy remainder under full sinusoidal-cavity
tracking. RF-off, degenerate, and invalid-parameter cases fail explicitly.

This is a local, nonresonant, conservative example. It does not include
radiation, acceleration, damping, spin, fringe fields, high-order action
detuning, resonance crossing, global mode tracking, or a validity guarantee
at finite large knob/amplitude excursions. Residuals validate the retained
Taylor coefficients, not those omitted effects.
