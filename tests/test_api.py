"""The public API inventory and its cross-module conventions."""

import inspect

import dsgelab as dsg


def test_root_exports_are_unique_and_defined():
    assert len(dsg.__all__) == len(set(dsg.__all__))
    assert all(hasattr(dsg, name) for name in dsg.__all__)


def test_public_iteration_limits_accept_max_iter():
    names = ["calibrate", "perfect_foresight", "extended_path", "discretionary_policy",
             "solve_markov_switching", "msv_solutions"]
    for name in names:
        signature = inspect.signature(getattr(dsg, name))
        assert "max_iter" in signature.parameters, name
        assert "maxiter" in signature.parameters, name
