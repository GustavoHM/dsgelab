"""compare_diagnostics.py: model diagnostics and the declared steady state, against Dynare.

Part 1, diagnose against model_diagnostics.

diagnostics/walras.mod, written here, is a model broken on purpose. The Euler equation is missing,
and in its place there is a resource constraint implied by the other two (Walras' law). Also, q
only appears one period ahead. Dynare runs model_diagnostics, and dsgelab reads the same file with
from_mod and runs diagnose at the steady_state_model point. What the two report is compared: the
variable absent in the current period, the number of collinear relationships, the variables and
the equations of each relationship, and the redundant equations of the dynamic Jacobian.

dsgelab's auxiliary variables (LAG1_k is k(-1)) are read under Dynare's name.

Part 2, the steady_state_model as the steady state (analytic=True), without Newton. The one in
Dynare's fs2000.mod has to give Dynare's point (out_import/fs2000.mat, from run_examples.sh). The
one in diagnostics/bad_steady.mod, wrong on purpose, has to be refused by both.

Part 3, homotopy. diagnostics/homotopy.mod, written here, has an initval that is good for
delta = 0.02 and the model with delta = 0.9, and homotopy_setup takes one to the other. Newton
alone fails. With the homotopy, dsgelab reaches the closed form and Dynare (homotopy_mode=3) the
same point, within Dynare's own tolerance.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import dsgelab as dsg
from dsgelab.dynare import from_mod

HERE = Path(__file__).resolve().parent
MOD = HERE / "diagnostics" / "walras.mod"


def _tools() -> tuple[str, Path] | None:
    """Octave and Dynare's matlab/ directory. OCTAVE and DYNARE_MATLAB take precedence."""
    octave = os.environ.get("OCTAVE") or shutil.which("octave") or "/opt/homebrew/bin/octave"
    candidates = [os.environ.get("DYNARE_MATLAB"), "/opt/homebrew/opt/dynare/lib/dynare/matlab",
                  "/opt/homebrew/lib/dynare/matlab", "/usr/local/opt/dynare/lib/dynare/matlab",
                  "/usr/lib/dynare/matlab"]
    matlab = next((Path(c) for c in candidates if c and Path(c).is_dir()), None)
    if not Path(octave).exists() or matlab is None:
        return None
    return octave, matlab


def dynare_report() -> str | None:
    if (tools := _tools()) is None:
        return None
    octave, matlab = tools
    with tempfile.TemporaryDirectory() as d:
        shutil.copy(MOD, d)
        out = subprocess.run([octave, "--no-gui", "--eval",
                              f"pkg load statistics datatypes; addpath '{matlab}'; "
                              "dynare walras noclearall nograph;"],
                             cwd=d, capture_output=True, text=True, timeout=600)
    return out.stdout


def parse(text: str) -> dict:
    absent = re.search(r"aren't present at the current period in the model:\n(\w+)", text)
    rel = re.search(r"There are (\d+) collinear relationship", text)
    static = text.split("Singularity in Static Jacobian")[1].split("MODEL_DIAGNOSTICS:  The presence")[0]
    vars_ = re.findall(r"^\s{4}(\w+)\s*$",
                       static.split("Collinear variables")[1].split("Collinear equations")[0], re.M)
    eqs = [int(x) for x in re.findall(r"Equation (\d+)", static.split("Collinear equations")[1])]
    dyn = text.split("Singularity in Dynamic Jacobian")[1].split("====")[2] \
        if "Singularity in Dynamic Jacobian" in text else ""
    red = re.search(r"There are (\d+) redundant equation", text)
    dyn_eqs = [int(x) for x in re.findall(r"Equation (\d+)", text.split("Singularity in Dynamic Jacobian")[1]
                                          .split("Singularity in Contemporaneous")[0])] \
        if "Singularity in Dynamic Jacobian" in text else []
    return dict(absent=[absent.group(1)] if absent else [], relations=int(rel.group(1)),
                variables=sorted(vars_), equations=sorted(eqs),
                redundant=int(red.group(1)) if red else 0, redundant_equations=sorted(dyn_eqs))


def main() -> int:
    text = dynare_report()
    if text is None:
        print("Octave/Dynare unavailable")
        return 1
    dy = parse(text)
    mf = from_mod(MOD)
    r = dsg.diagnose(mf.model, mf.params, mf.full_guess())
    ours = dict(absent=r["absent_current"], relations=r["relations"],
                variables=sorted({mf.name_of(v) for v in r["collinear_variables"][0]}),
                equations=sorted(i for i, _ in r["collinear_equations"][0]),
                redundant=r["redundant_equations"],
                redundant_equations=sorted(i for i, _ in r["redundant_groups"][0]))
    ok = True
    for k in dy:
        same = dy[k] == ours[k]
        ok &= same
        print(f"  {k:<20} dsgelab {str(ours[k]):<28} Dynare {str(dy[k]):<28} {'OK' if same else 'DIVERGE'}")
    ok &= check_closed_forms()
    ok &= check_homotopy()
    return 0 if ok else 1


def check_homotopy() -> bool:
    import numpy as np
    from dsgelab.steady import SteadyStateError  # noqa: F401
    print("\nhomotopy (homotopy_setup)")
    mf = from_mod(HERE / "diagnostics" / "homotopy.mod")
    a, b, d = (mf.params[x] for x in ("alpha", "beta", "delta"))
    k = (a / (1 / b - 1 + d)) ** (1 / (1 - a))
    exact = dict(k=k, c=k ** a - d * k)
    try:
        mf.steady_state()
        direct = "converges (unexpected)"
    except Exception:
        direct = "fails"
    ss = mf.steady_state(homotopy=mf.homotopy)
    err = max(abs(ss.values[v] / exact[v] - 1) for v in exact)
    print(f"  Newton alone: {direct}; with homotopy, against the closed form  {err:.1e}")
    text = _run_dynare(HERE / "diagnostics" / "homotopy.mod")
    dy = {}
    if text:
        for v in exact:
            m = re.search(rf"^\s*{v}\s+([-\d.eE+]+)\s*$", text.split("STEADY-STATE RESULTS")[-1], re.M)
            if m:
                dy[v] = float(m.group(1))
    if len(dy) != len(exact):
        print("  Dynare: no steady state in the log")
        return False
    errd = max(abs(dy[v] / exact[v] - 1) for v in exact)
    print(f"  Dynare, against the closed form  {errd:.1e}  (what the log prints, with 6 digits)")
    return bool(err < 1e-12 and errd < 1e-5 and direct == "fails" and np.isfinite(errd))


def check_closed_forms() -> bool:
    import numpy as np
    from scipy.io import loadmat
    from dsgelab.steady import SteadyStateError
    print("\nsteady_state_model as the steady state")
    ok = True
    f = HERE / "out_import" / "fs2000.mat"
    if f.exists():
        ex = loadmat(f, squeeze_me=True, struct_as_record=False)["ex"]
        ys = dict(zip([str(x).strip() for x in np.atleast_1d(ex.endo_names)], np.atleast_1d(ex.ys)))
        mf = from_mod(HERE / "mods" / "examples" / "fs2000.mod")
        ss = mf.steady_state(analytic=True)
        d = max(abs(ss.values[v] - ys[v]) for v in mf.endogenous)
        print(f"  fs2000: closed form against Dynare's s.s.  {d:.1e}   residual {ss.residual:.1e}")
        ok &= d < 1e-12
    else:
        print("  fs2000: no out_import/fs2000.mat, run run_examples.sh")
        ok = False
    try:
        from_mod(HERE / "diagnostics" / "bad_steady.mod").steady_state(analytic=True)
        print("  bad_steady: ACCEPTED, should have been refused")
        ok = False
    except SteadyStateError as e:
        print(f"  bad_steady: refused ({str(e)[:90]}...)")
    text = _run_dynare(HERE / "diagnostics" / "bad_steady.mod")
    refused = text is not None and "did not compute the steady state" in text
    print(f"  bad_steady in Dynare: {'refused' if refused else 'ACCEPTED'}")
    return ok and refused


def _run_dynare(mod: Path) -> str | None:
    if (tools := _tools()) is None:
        return None
    octave, matlab = tools
    with tempfile.TemporaryDirectory() as d:
        shutil.copy(mod, d)
        out = subprocess.run([octave, "--no-gui", "--eval",
                              f"pkg load statistics datatypes; addpath '{matlab}'; "
                              f"dynare {mod.stem} noclearall nograph;"],
                             cwd=d, capture_output=True, text=True, timeout=600)
    return out.stdout + out.stderr


if __name__ == "__main__":
    sys.exit(main())
