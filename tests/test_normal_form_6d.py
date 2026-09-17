"""Independent physical and algebraic checks of the coupled 6D example."""
from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pytest

import madng_tpsa_test as mt

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples' / 'parametric_normal_form_6d.py'


@pytest.fixture(scope='module')
def example():
    return runpy.run_path(str(EXAMPLE))


@pytest.fixture(scope='module')
def result(example):
    return example['analyze']()


def test_all_three_modes_are_coupled_and_canonical(result, example):
    a0 = example['matrix_nominal'](result.a).real
    # Coupling is present between every pair of physical planes.
    for i, j in ((0, 1), (0, 2), (1, 2)):
        assert np.linalg.norm(a0[2*i:2*i+2, 2*j:2*j+2]) > 1e-3
    tunes = [x.nominal().real for x in result.tunes]
    assert all(0 < q < 0.5 for q in tunes)
    assert len(set(round(q, 5) for q in tunes)) == 3
    for name, error in result.errors.items():
        if name != 'minimum_nonresonant_denominator':
            assert error < 1e-10, (name, error)
    assert result.errors['minimum_nonresonant_denominator'] > 1e-3
    assert all(result.h[i].degree(2).norm() > 1e-5 for i in range(6))
    assert all(result.h[i].degree(3).norm() > 1e-5 for i in range(6))


def test_native_exports_preserve_parameter_coefficients(result):
    native = result.native()
    d = native['descriptor']
    assert d.n_variables == 6
    assert d.n_parameters == len(result.ring.names)
    assert isinstance(native['H'], mt.CTPSAMap)
    assert isinstance(native['A'][0][0], mt.CTPSA)
    for j, name in enumerate(result.ring.names):
        p = tuple(int(k == j) for k in range(len(result.ring.names)))
        for i in range(6):
            expected = result.orbit[i].data.get(p)
            value = expected.constant_term if expected is not None else 0
            assert native['closed_orbit'][i][(0,)*6+p] == pytest.approx(value, abs=1e-13)
    phi = tuple(int(name == 'phirf') for name in result.ring.names)
    assert native['closed_orbit'][4][(0,)*6+phi] == pytest.approx(-1/1.1, abs=1e-11)


@pytest.mark.parametrize('knob', ['kf', 'kxy', 'vrf'])
def test_parameter_tune_and_detuning_derivatives_against_finite_differences(result, example, knob):
    h = 2e-6
    nominal = example['DEFAULTS'][knob]
    plus = example['analyze'](parameter_order=0, parameters=(), nominal={knob: nominal+h})
    minus = example['analyze'](parameter_order=0, parameters=(), nominal={knob: nominal-h})
    p = tuple(int(name == knob) for name in result.ring.names)
    for j in range(3):
        expected = result.tunes[j].data[p].constant_term.real
        measured = (plus.tunes[j].nominal()-minus.tunes[j].nominal()).real/(2*h)
        assert measured == pytest.approx(expected, rel=3e-6, abs=2e-9)
        for k in range(3):
            term = result.detuning[j][k].data.get(p)
            expected = term.constant_term.real if term is not None else 0.0
            measured = (plus.detuning[j][k].nominal()-minus.detuning[j][k].nominal()).real/(2*h)
            assert measured == pytest.approx(expected, rel=3e-5, abs=2e-8)


def test_square_and_mixed_parameter_coefficients(example):
    result = example['analyze'](parameter_order=2, parameters=('ks', 'vrf'))
    assert result.errors['symplectic_H'] < 1e-9
    h = 2e-4
    nominal = example['DEFAULTS']['ks']
    plus = example['analyze'](parameter_order=0, parameters=(), nominal={'ks': nominal+h})
    minus = example['analyze'](parameter_order=0, parameters=(), nominal={'ks': nominal-h})
    center = result.detuning[0][0].nominal().real
    square = result.detuning[0][0].data[(2, 0)].constant_term.real
    assert abs(square) > 1e-3
    measured = ((plus.detuning[0][0].nominal()+minus.detuning[0][0].nominal()).real-2*center)/(2*h*h)
    # Stored Taylor coefficient is derivative / 2!, not the raw second derivative.
    assert measured == pytest.approx(square, rel=2e-6, abs=2e-8)
    mixed = result.detuning[0][0].data[(1, 1)].constant_term.real
    assert abs(mixed) > 1e-5
    vp = example['analyze'](parameters=('ks',), nominal={'vrf': example['DEFAULTS']['vrf']+h})
    vm = example['analyze'](parameters=('ks',), nominal={'vrf': example['DEFAULTS']['vrf']-h})
    measured = (vp.detuning[0][0].data[(1,)].constant_term-vm.detuning[0][0].data[(1,)].constant_term).real/(2*h)
    assert measured == pytest.approx(mixed, rel=5e-5, abs=2e-7)


def test_physical_tracking_conjugacy_has_fourth_order_remainder(result, example):
    native = result.native()
    b, n = native['nonlinear_A'], native['N']
    direction = np.array([0.4+0.2j, 0.4-0.2j, 0.3-0.15j, 0.3+0.15j, 0.2+0.1j, 0.2-0.1j])
    parameters = [0.0]*len(result.ring.names)
    errors = []
    for radius in (0.02, 0.01, 0.005):
        w = radius*direction
        physical = np.array(b.evaluate([*w, *parameters]))
        assert np.max(np.abs(physical.imag)) < 1e-12
        tracked = np.array(example['track'](physical.real, example['DEFAULTS']))
        normal = n.evaluate([*w, *parameters])
        reconstructed = np.array(b.evaluate([*normal, *parameters]))
        errors.append(np.max(np.abs(tracked-reconstructed)))
    # Independent numeric tracking uses the full sinusoidal cavity, not its TPSA.
    assert all(12 < a/b < 20 for a, b in zip(errors, errors[1:])), errors


def test_native_nominal_poisson_brackets(result):
    # Independent direct C Poisson brackets, rather than the Jet derivative helper.
    h = [f.data[result.ring.zero] for f in result.h]
    for i in range(6):
        for j in range(i+1, 6):
            bracket = h[i].poisson_bracket(h[j], n_variables=6)
            expected = 1.0 if i % 2 == 0 and j == i+1 else 0.0
            error = bracket-expected
            # Complex-coordinate Poisson tensor is -i*J; the common factor cancels.
            for degree in range(3):
                assert max((abs(c) for _, c in error.get_order(degree).coefficients()), default=0) < 1e-11


def test_rf_off_and_invalid_parameters_fail_explicitly(example):
    with pytest.raises(ValueError):
        example['analyze'](nominal={'vrf': 0.0})
    with pytest.raises(ValueError):
        example['analyze'](parameters=('ks', 'ks'))
    with pytest.raises(ValueError):
        example['analyze'](parameter_order=3)


def test_native_composition_keeps_parameter_identity(example):
    d = mt.descriptor(6, 4, n_parameters=1, parameter_order=1)
    identity = mt.CTPSAMap.identity(d)
    parameter = mt.CTPSA.constant(d)
    parameter[(0,)*6 + (1,)] = 1
    outer = mt.CTPSAMap([identity[0]*(1+parameter), *identity[1:]])
    composed = example['compose_native'](outer, identity)
    assert composed[0][(1,0,0,0,0,0,1)] == pytest.approx(1.0)
    assert composed.evaluate([0.2,0,0,0,0,0,0.3])[0] == pytest.approx(0.26)
