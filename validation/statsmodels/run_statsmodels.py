"""run_statsmodels.py: the references of compare_kim.py, from statsmodels' Markov-switching models.

Reads ../out_kim/input.json, written by ``compare_kim.py --prepare``, and writes
../out_kim/statsmodels.json. Two series, each simulated by dsgelab from a Markov-switching model:

  hamilton  y_t = mu[s_t] + sigma[s_t] e_t, statsmodels' MarkovRegression with a switching
            constant and a switching variance;
  ar        y_t - mu[s_t] = phi (y_{t-1} - mu[s_{t-1}]) + sigma e_t, MarkovAutoregression of
            order 1 with a switching constant (Hamilton, 1989).

For each, at the parameters of the model that simulated the data: the log likelihood, the filtered
and the smoothed regime probabilities. For the first, also the maximum-likelihood estimate, from
statsmodels' fit started at those parameters.

Run with the environment of setup.sh:
    validation/statsmodels/.venv/bin/python validation/statsmodels/run_statsmodels.py
"""
from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

OUT = Path(__file__).resolve().parent.parent / "out_kim"


def main() -> None:
    spec = json.loads((OUT / "input.json").read_text())
    out = {"statsmodels": version("statsmodels")}

    h = spec["hamilton"]
    y = np.asarray(h["y"])
    mod = MarkovRegression(y, k_regimes=2, trend="c", switching_variance=True)
    params = np.asarray(h["params"])        # p[0->0], p[1->0], const[0], const[1], sigma2[0], sigma2[1]
    res = mod.smooth(params)
    fit = mod.fit(start_params=params, disp=False, maxiter=500)
    out["hamilton"] = dict(
        param_names=list(mod.param_names), loglik=float(mod.loglike(params)),
        filtered=res.filtered_marginal_probabilities[:, 0].tolist(),
        smoothed=res.smoothed_marginal_probabilities[:, 0].tolist(),
        mle=fit.params.tolist(), mle_loglik=float(fit.llf))

    a = spec["ar"]
    y = np.asarray(a["y"])
    mod = MarkovAutoregression(y, k_regimes=2, order=1, trend="c", switching_ar=False,
                               switching_variance=False)
    params = np.asarray(a["params"])        # p[0->0], p[1->0], const[0], const[1], sigma2, ar.L1
    res = mod.smooth(params)
    out["ar"] = dict(
        param_names=list(mod.param_names), loglik=float(mod.loglike(params)),
        filtered=res.filtered_marginal_probabilities[:, 0].tolist(),
        smoothed=res.smoothed_marginal_probabilities[:, 0].tolist())

    (OUT / "statsmodels.json").write_text(json.dumps(out, indent=1))
    print("statsmodels", out["statsmodels"], "->", OUT / "statsmodels.json")


if __name__ == "__main__":
    main()
