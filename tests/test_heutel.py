"""Validate the port against original MATLAB algebra and independent Dynare IRFs."""
import json
from pathlib import Path

import numpy as np
import pytest

from dsgelab.examples import heutel

REFERENCE = json.loads((Path(__file__).resolve().parents[1]
                        / 'validation/heutel/reference.json').read_text())


def params(variant):
    return heutel.Params() if variant == 'planner' else heutel.Params.source(variant)


@pytest.mark.parametrize('variant', heutel.VARIANTS)
def test_original_octave_algebra_and_dynare_dynamics(variant):
    p = params(variant)
    ref = REFERENCE[variant]
    names, h, ss = heutel.structural_system(variant, p)
    assert list(names) == ref['names']
    np.testing.assert_allclose([ss[n] for n in names], ref['levels'], rtol=1e-9, atol=1e-8)
    np.testing.assert_allclose(h, ref['h'], rtol=1e-8, atol=1e-10)
    sol = heutel.solve(variant, p)
    irf = sol.irf('eps_a', size=.01, periods=100, percent=False)
    for name in names:
        np.testing.assert_allclose(irf[name], ref['irfs'][name], rtol=1e-7, atol=1e-10)
    mom = sol.moments(cov={'eps_a': .01})
    np.testing.assert_allclose([mom['std'][sol.model.position(n)] for n in names],
                               ref['std'], rtol=1e-7, atol=1e-10)
    assert sol.klein.shock_residual < 1e-12


@pytest.mark.parametrize('variant', heutel.VARIANTS)
def test_identities_timing_and_policy_feedback(variant):
    p = params(variant)
    ss = heutel.steady_levels(variant, p)
    sol = heutel.solve(variant, p)
    irf = sol.irf('eps_a', size=.01, periods=100, percent=False)
    np.testing.assert_allclose(irf['a'], .01*p.rho**np.arange(100), atol=1e-12)
    np.testing.assert_allclose(ss['c']*irf['c'] + ss['i']*irf['i']
                               + ss['z']*irf['z'], ss['y']*irf['y'], atol=1e-12)
    np.testing.assert_allclose(irf['k'][1:], (1-p.delta)*irf['k'][:-1]
                               + p.delta*irf['i'][1:], atol=1e-12)
    np.testing.assert_allclose(irf['x'][1:], p.eta*irf['x'][:-1]
                               + (1-p.eta)*irf['e'][1:], atol=1e-12)
    if variant.endswith('asym'):
        policy = 'q' if variant == 'quota_asym' else 'tau'
        assert abs(irf[policy][0]) < 1e-12
        # This would fail if state_2 propagated the full-information state,
        # as in the original MATLAB script's second simulation loop.
        np.testing.assert_allclose(irf[policy][1:], p.policy_slope*irf['y'][:-1], atol=1e-12)
        if policy == 'q':
            np.testing.assert_allclose(irf['e'], irf['q'], atol=1e-12)


def test_literal_planner_defaults_are_not_silently_changed():
    p = heutel.Params.source('planner')
    assert p.theta2 == .1 and p.gamma == .8
    with pytest.raises(heutel.SteadyStateError, match='no admissible converged'):
        heutel.build('planner')


def test_zero_ramsey_multiplier_has_an_absolute_coordinate():
    sol = heutel.solve('tax')
    assert sol.klein.n_stable == 4
    assert abs(heutel.steady_levels()['lambda']) < 1e-10
    levels = heutel.to_levels({'lambda': np.array([.1]), 'a': np.array([1.0])})
    assert levels['lambda'][0] == pytest.approx(.001, abs=1e-10)
    assert levels['a'][0] == pytest.approx(1.01)


def test_changed_calibration_requires_rebuilding():
    model, p, guess = heutel.build('tax_asym')
    from dataclasses import replace
    from dsgelab import solve
    with pytest.raises(ValueError, match='rebuild'):
        solve(model, replace(p, rho=.9), guess=guess)


@pytest.mark.parametrize('variant', ['invalid', 'quota'])
def test_invalid_variant(variant):
    with pytest.raises(ValueError, match='variant must'):
        heutel.build(variant)
