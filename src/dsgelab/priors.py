"""Prior distributions, parametrised as Dynare's ``estimated_params`` parametrises them.

Each prior is given by its MEAN and STANDARD DEVIATION, plus, where Dynare takes
them, a third and fourth number: the bounds of a generalised beta or a uniform,
the shift of a gamma or an inverse gamma. The log densities are Dynare's, with
the same normalising constants, so that a log posterior computed here can be
compared with Dynare's number for number.

=================  =========================================  ===================
``beta``           generalised beta on ``[p3, p4]`` (0, 1)     Dynare ``beta_pdf``
``gamma``          gamma shifted by ``p3`` (0)                 ``gamma_pdf``
``normal``         normal                                      ``normal_pdf``
``inv_gamma``      inverse gamma type 1, for a standard        ``inv_gamma_pdf``
                   deviation; ``sd = inf`` gives ``nu = 2``    (``inv_gamma1_pdf``)
``inv_gamma2``     inverse gamma type 2, for a variance        ``inv_gamma2_pdf``
``uniform``        uniform on ``[p3, p4]``; or on the interval ``uniform_pdf``
                   with the given mean and standard deviation
=================  =========================================  ===================
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import betaln, gammaln

__all__ = ["Prior", "prior"]


@dataclass
class Prior:
    """A prior: shape, the two parameters of the density, and its support."""

    shape: str
    mean: float
    sd: float
    a: float            # shape-specific first parameter
    b: float            # shape-specific second parameter
    lower: float
    upper: float

    def logpdf(self, x: float) -> float:
        x = float(x)
        if not (self.lower <= x <= self.upper) or (x == self.lower and self.shape != "normal"
                                                  and self.shape != "uniform"):
            return -np.inf
        s = self.shape
        if s == "normal":
            return float(-0.5 * math.log(2 * math.pi) - math.log(self.sd)
                         - 0.5 * ((x - self.mean) / self.sd) ** 2)
        if s == "beta":
            aa, bb = self.lower, self.upper
            if x >= bb:
                return -np.inf
            return float(-betaln(self.a, self.b) - (self.a + self.b - 1) * math.log(bb - aa)
                         + (self.a - 1) * math.log(x - aa) + (self.b - 1) * math.log(bb - x))
        if s == "gamma":
            y = x - self.lower
            return float(-gammaln(self.a) - self.a * math.log(self.b)
                         + (self.a - 1) * math.log(y) - y / self.b)
        if s == "inv_gamma":
            # Dynare lpdfig1: s = a, nu = b
            y = x - self.lower
            nu, ss = self.b, self.a
            return float(math.log(2) - gammaln(nu / 2) + (nu / 2) * math.log(ss / 2)
                         - (nu + 1) * math.log(y) - ss / (2 * y * y))
        if s == "inv_gamma2":
            y = x - self.lower
            nu, ss = self.b, self.a
            return float(-gammaln(nu / 2) + (nu / 2) * math.log(ss / 2)
                         - (nu / 2 + 1) * math.log(y) - ss / (2 * y))
        if s == "uniform":
            return float(-math.log(self.upper - self.lower))
        raise ValueError(f"unknown prior shape {s!r}")

    def sample(self, rng: np.random.Generator,
               size: int | tuple[int, ...] | None = None) -> float | np.ndarray:
        s = self.shape
        if s == "normal":
            return rng.normal(self.mean, self.sd, size)
        if s == "beta":
            return self.lower + (self.upper - self.lower) * rng.beta(self.a, self.b, size)
        if s == "gamma":
            return self.lower + rng.gamma(self.a, self.b, size)
        if s == "inv_gamma":
            # sigma^2 ~ IG2(s, nu): s / chi2(nu)
            return self.lower + np.sqrt(self.a / rng.chisquare(self.b, size))
        if s == "inv_gamma2":
            return self.lower + self.a / rng.chisquare(self.b, size)
        if s == "uniform":
            return rng.uniform(self.lower, self.upper, size)
        raise ValueError(s)


_ALIASES = {"beta_pdf": "beta", "gamma_pdf": "gamma", "normal_pdf": "normal",
            "inv_gamma_pdf": "inv_gamma", "inv_gamma1_pdf": "inv_gamma",
            "inv_gamma2_pdf": "inv_gamma2", "uniform_pdf": "uniform",
            "invgamma": "inv_gamma", "inverse_gamma": "inv_gamma"}


def _ig1_from_moments(mu, sigma):
    """Dynare's inverse_gamma_specification, type 1: (s, nu) from mean and sd."""
    if not np.isfinite(sigma):
        nu = 2.0
        return 2 * mu ** 2 / math.pi, nu
    sigma2 = sigma ** 2
    mu2 = mu ** 2
    # mean^2 / variance pins down nu through a ratio of gamma functions
    def err(nu):
        s = 2 * (mu / math.exp(gammaln((nu - 1) / 2) - gammaln(nu / 2))) ** 2
        return s / (nu - 2) - mu2 - sigma2
    lo, hi = 2.0 + 1e-10, 2.0 + 1.0
    while err(hi) > 0:
        hi = 2 + 2 * (hi - 2)
        if hi > 1e8:
            raise ValueError("inverse gamma: cannot match the moments")
    nu = brentq(err, lo, hi, xtol=1e-14, rtol=1e-14)
    s = 2 * (mu / math.exp(gammaln((nu - 1) / 2) - gammaln(nu / 2))) ** 2
    return s, nu


def prior(shape: str, mean: float, sd: float, p3: float | None = None,
          p4: float | None = None) -> Prior:
    """A prior in Dynare's terms: shape, mean, standard deviation, and p3/p4."""
    s = _ALIASES.get(shape, shape)
    mean, sd = float(mean), float(sd)
    if s == "normal":
        return Prior("normal", mean, sd, mean, sd, -np.inf, np.inf)
    if s == "beta":
        aa = 0.0 if p3 is None or not np.isfinite(p3) else float(p3)
        bb = 1.0 if p4 is None or not np.isfinite(p4) else float(p4)
        mu = (mean - aa) / (bb - aa)
        v = (sd / (bb - aa)) ** 2
        a = (1 - mu) * mu ** 2 / v - mu
        b = a * (1 / mu - 1)
        if a <= 0 or b <= 0:
            raise ValueError(f"beta prior: mean {mean} and sd {sd} are not attainable")
        return Prior("beta", mean, sd, a, b, aa, bb)
    if s == "gamma":
        shift = 0.0 if p3 is None or not np.isfinite(p3) else float(p3)
        m = mean - shift
        a = m ** 2 / sd ** 2
        b = sd ** 2 / m
        return Prior("gamma", mean, sd, a, b, shift, np.inf)
    if s == "inv_gamma":
        shift = 0.0 if p3 is None or not np.isfinite(p3) else float(p3)
        ss, nu = _ig1_from_moments(mean - shift, sd)
        return Prior("inv_gamma", mean, sd, ss, nu, shift, np.inf)
    if s == "inv_gamma2":
        shift = 0.0 if p3 is None or not np.isfinite(p3) else float(p3)
        m = mean - shift
        if np.isfinite(sd):
            nu = 2 * (2 + m ** 2 / sd ** 2)
            ss = 2 * m * (1 + m ** 2 / sd ** 2)
        else:
            nu, ss = 4.0, 2 * m
        return Prior("inv_gamma2", mean, sd, ss, nu, shift, np.inf)
    if s == "uniform":
        if p3 is not None and p4 is not None and np.isfinite(p3) and np.isfinite(p4):
            lo, hi = float(p3), float(p4)
        else:
            half = math.sqrt(3) * sd
            lo, hi = mean - half, mean + half
        return Prior("uniform", (lo + hi) / 2, (hi - lo) / math.sqrt(12), lo, hi, lo, hi)
    raise ValueError(f"unknown prior shape {shape!r}")
