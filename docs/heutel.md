# Heutel: port of the MATLAB scripts to dsgelab

The module `dsgelab.examples.heutel` ports the first-order systems of the paper's six
MATLAB replication files, whose public copy `validation/heutel/fetch.sh`
downloads (see the end of this page).
dsgelab's Klein/QZ takes the place of the Anderson–Moore solver, which the supplied files do not include.
The installed package depends on NumPy and SciPy alone. MATLAB, Octave and Dynare are
needed only to repeat the independent validation.

Source: Garth Heutel (2012), *How should environmental policy respond to business
cycles? Optimal policy under persistent productivity shocks*, Review of Economic
Dynamics 15, 244–264, DOI 10.1016/j.red.2011.05.002. The scripts come with a README dated
23/05/2011 and carry the earlier title “Optimal Environmental Policy Under Economic
Fluctuations”. The page numbers cited here are those of the published version of the paper, whose
sha256 is in `validation/heutel/paper-source.json`.

## Usage

```python
from dsgelab.examples import heutel

# Once per machine: download, check and translate Heutel's MATLAB files.
heutel.fetch()

# Optimal tax, full information; parameters of the tax script.
sol = heutel.solve("tax")
irf = sol.irf("eps_a", size=0.01, periods=100)   # per cent; a 1% shock
ss = heutel.steady_levels("tax")
levels = heutel.to_levels(irf, "tax")

# Rules with lagged information; original slopes 1.4151 and 0.6580.
tax_asym = heutel.solve("tax_asym")
quota_asym = heutel.solve("quota_asym")

# Planner under the Table 2 calibration: an EXPLICIT choice.
planner = heutel.solve("planner", params=heutel.Params())

# Simulations use fractional deviations, not multiplied by 100.
sim = sol.simulate(periods=1000, cov={"eps_a": 0.007}, seed=42)
sim_levels = heutel.to_levels(sim, "tax", percent=False)
moments = sol.moments(cov={"eps_a": 0.007})
```

The standard interface also works.

```python
import dsgelab as dk
model, params, guess = heutel.build("tax_asym")
sol = dk.solve(model, params, guess=guess)
```

`build` declares the system **already linearised**, in deviations with a zero steady
state. `sol.steady` and `sol.irf(levels=True)` therefore refer to the coordinates of that
system, and the economic levels are recovered with `steady_levels` and `heutel.to_levels`.
After changing parameters, rebuild the model, because the callback rejects parameters
other than those that generated its matrix.

## Variants and calibration

| Variant | Script | Default calibration |
|---|---|---|
| `planner` | `SolveSystem.m` + `steadystate.m` | θ₂=0.1; γ=0.8, as in the file |
| `tax` | `SolveSystem_tax.m` + `steadystate_tax.m` | θ₂=2.8; γ=0.304 |
| `tax_asym` | `SolveSystem_tax_AsymInfo.m` | as above; τ̂ₜ=1.4151 ŷₜ₋₁ |
| `quota_asym` | `SolveSystem_quota_AsymInfo.m` | as above; q̂ₜ=0.6580 ŷₜ₋₁ |

`Params()` holds the calibration of the tax scripts. `Params.source(variant)`
returns the defaults of each file, including the planner's different ones.

The planner's literal calibration did not produce an admissible steady state
in the procedures that were run. The bounded solver ends at the boundary
μ≈1, with a maximum residual of ≈0.00965. Instead of accepting the output of `fsolve` without
checking consumption, emissions, μ and the residuals, the port raises `SteadyStateError`.
A failure in these procedures does not prove that no steady state exists globally. For the
Table 2 baseline, `heutel.solve("planner", heutel.Params())` solves while keeping the
script's equations, and the numerical comparisons below use that explicit choice.

## Operational corrections and coordinates

- The original `k` is end-of-period capital, and `x` is the contemporaneous stock.
  The auxiliary states `lag_k`, `lag_x` and, where needed, `lag_lambda` or
  `lag_y` hold the lagged values. Current productivity is `a`.
- The shock follows the dsgelab convention, with εₜ₊₁ in the transition of `a`. The IRF
  keeps the scripts' impact period, â₀=0.01.
- Under the optimal tax, the steady-state λ is approximately zero, and log-linearising
  this multiplier makes the matrix nearly singular. The port therefore uses **absolute
  deviations of λ** and evaluates their coefficients per unit of multiplier, so that
  nothing is divided by a number close to zero. The other variables use proportional
  deviations, and ω may have a negative steady-state level. The option `percent=True`
  also multiplies the absolute deviation of λ by 100, which does not turn it into an
  economic percentage change.
- The asymmetric scripts propagate `state_2(:,t)` from `state(:,t-1)` instead of from
  their own path `state_2(:,t-1)`, in both the IRF and the simulation.
  dsgelab follows the path of the chosen regime, and the tests check the policy rule
  in every period.
- `structural_system` returns the raw H matrix, in the file's order and coordinates,
  before the normalisation of λ that `build` performs.

## Check against the supplied paper

The PDF was checked at pp. 249–251 and 258 (equations, calibration and sensitivity).
The literal port preserves the following material differences.

1. Table 2, p. 251, gives θ₂=2.8 and 1−γ=0.696. Page 258 discusses the values
   θ₂=0.1 and 1−γ=0.2 themselves as a sensitivity case. The planner file therefore
   appears to have been saved with that sensitivity configured.
2. Table 2 and the six scripts use β=0.98267, whereas the text on p. 250 prints 0.98627.
   Following the table and the code, the port does not settle the discrepancy by assumption.
3. The text, pp. 250–251, holds foreign emissions constant at three times
   steady-state domestic emissions. With x̄=4 ē/(1−η), this implies
   x̂ₜ=η x̂ₜ₋₁+((1−η)/4) êₜ. The scripts' matrices use `(1−η)` as the
   coefficient of êₜ, four times that value. The original matrix was kept
   and tested as it stands, and it should not be presented as an exact implementation
   of that assumption in the text.
4. The scripts scale the damage coefficients by 5.3024. Footnote 20,
   p. 251, explains that carbon units have to be made compatible with
   the units of the model. `Params()` already holds the scaled d₁ and d₂.

Agreement with Dynare thus validates the transcription and the solution of the supplied
systems. It does not establish that every published equation, the literal sensitivity
calibration or every figure in the paper has been reproduced.

## Independent validation

`validation/heutel/validate.py` runs the prefixes of the original scripts in
Octave up to the construction of H, without depending on AIM. It exports levels and matrices,
generates linear models from those matrices in Dynare and compares the steady state,
the 100-period IRFs and the theoretical standard deviations of every variable.
For the tax, the change of coordinate of λ is also computed directly in
MATLAB from the original coefficients. The reference paths use a
shock size and standard deviation of 0.01 in both solvers, with no HP filter.

| Variant | Maximum error: levels | H matrix | IRFs | Standard deviations |
|---|---:|---:|---:|---:|
| Planner, common calibration | 5.14e-9 | 2.74e-12 | 3.05e-12 | 1.61e-11 |
| Tax | 9.55e-11 | 5.92e-13 | 8.36e-13 | 5.83e-12 |
| Tax, asymmetric information | 9.55e-11 | 4.99e-14 | 6.00e-16 | 1.07e-14 |
| Quota, asymmetric information | 9.55e-11 | 4.99e-14 | 3.50e-13 | 3.55e-13 |

Both commands run from the root of the repository.

```bash
pytest -q
python validation/heutel/validate.py
```

The first uses Octave/Dynare references stored in `reference.json` and needs neither
program installed. The second requires Octave and Dynare and reads the variables
`OCTAVE` and `DYNARE_MATLAB`. The hashes of the supplied files are in `source-sha256.json`.
The port's scalar algebra is not distributed with the package, because it is a translation of
files that declare no licence. `heutel.fetch()` generates it on the user's machine.
The function downloads the public copy (or reads the `.zip` or folder it is given), refuses
any file whose SHA-256 differs from the recorded one, and translates scalar assignments and
the coefficients of the selected regions and nothing else, without executing any MATLAB text.
The translated module is written to `~/.cache/dsgelab/heutel`, or to `DSGELAB_HEUTEL_DIR`, and
is loaded only if its own SHA-256 matches that of the validated translation. Without it, the
Heutel tests are skipped and `heutel.solve` raises `HeutelSourceMissing`, which says what to run.

## Limitations

This port covers the steady state, first-order dynamics, IRFs, simulations and moments
of the MATLAB systems. The Stata regressions are not ported. Nor does the port estimate
the optimal slope of the rules, since the supplied files contain only a commented-out call
to the function `IRFdiff`, which is missing from them. Random series that the originals drew
without a seed are not reproduced either. The difference between the original asymmetric
paths and the corrected ones is intentional and is checked through the policy equation.

A utility function applied to the linear path does not amount to a second-order
welfare analysis. Do not use `solve_second_order` on this linear model as if it
provided that analysis, which requires declaring and checking the full nonlinear
system and reconciling its differences with the paper.

The original files remain attributed to Garth Heutel and are not included as MIT-licensed
dsgelab code. A public copy is held in Georgia State University's institutional repository
(scholarworks.gsu.edu, item `9a01aa68-89c0-4e66-b7bd-9f09606d259c`).
`bash validation/heutel/fetch.sh` downloads the archive, checks its sha256 and extracts the
`.m` files into `validation/heutel/original`, and `python validation/heutel/validate.py` then
repeats the validation. The six `.m` files of the public copy are byte-for-byte identical to
those the validation used, and `source-sha256.json` records their hashes. The public
`ReadMe.txt` is an earlier version (4 March 2011) that describes only the MATLAB files.
The version of 23 May 2011, which accompanied the supplied files, also describes the
Stata ones.
