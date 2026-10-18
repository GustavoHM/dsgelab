"""Projection against the closed-form stochastic growth policy."""

import numpy as np

import dsgelab as dsg
from dsgelab.examples import rbc


def test_chebyshev_projection_recovers_the_closed_form_policy():
    model, params, guess = rbc.build()
    sol = dsg.solve(model, params, guess=guess)
    k_ss = sol.steady.values["k"]
    proj = dsg.solve_projection(
        model, params, bounds=[(0.7 * k_ss, 1.4 * k_ss), (0.92, 1.08)],
        guess=guess, order=4, cov={"eps_a": 0.007}, n_quad=5, steady=sol.steady,
    )
    assert proj.success
    assert proj.residual < 1e-5
    rng = np.random.default_rng(0)
    k = rng.uniform(0.8 * k_ss, 1.3 * k_ss, size=12)
    a = rng.uniform(0.94, 1.06, size=12)
    y = params.alpha
    for ki, ai in zip(k, a):
        c = proj.jumps([ki, ai])[0]
        exact_c = (1.0 - params.alpha * params.beta) * ai * ki ** y
        exact_k = params.alpha * params.beta * ai * ki ** y
        assert abs(c - exact_c) / exact_c < 5e-3
        # next capital from the resource constraint, which the collocation enforces ex post
        kn = (1.0 - params.delta) * ki + ai * ki ** y - c
        assert abs(kn - exact_k) / exact_k < 5e-3
