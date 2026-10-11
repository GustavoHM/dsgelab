"""compare_diagnostics.py — diagnóstico do modelo e estado estacionário declarado, contra o Dynare.

Parte 1, diagnose contra model_diagnostics.

diagnostics/walras.mod, escrito aqui, é um modelo quebrado de propósito: falta a equação de
Euler, e no lugar dela está uma restrição de recursos implicada pelas outras duas (a lei de
Walras); e q só aparece um período adiante. O Dynare roda o model_diagnostics; o dsgelab lê o
mesmo arquivo por from_mod e roda diagnose no ponto do steady_state_model. Compara-se o que os
dois apontam: a variável ausente no período corrente, o número de relações colineares, as
variáveis e as equações de cada relação, e as equações redundantes da jacobiana dinâmica.

As variáveis auxiliares do dsgelab (LAG1_k é k(-1)) são lidas pelo nome do Dynare.

Parte 2, o steady_state_model como estado estacionário (analytic=True), sem Newton. O do fs2000.mod
do Dynare tem de dar o ponto do Dynare (out_import/fs2000.mat, de run_examples.sh); o de
diagnostics/bad_steady.mod, errado de propósito, tem de ser recusado pelos dois.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import dsgelab as dk
from dsgelab.dynare import from_mod

HERE = Path(__file__).resolve().parent
MOD = HERE / "diagnostics" / "walras.mod"


def dynare_report() -> str | None:
    octave = shutil.which("octave") or "/opt/homebrew/bin/octave"
    matlab = Path("/opt/homebrew/opt/dynare/lib/dynare/matlab")
    if not Path(octave).exists() or not matlab.exists():
        return None
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
        print("Octave/Dynare indisponíveis")
        return 1
    dy = parse(text)
    mf = from_mod(MOD)
    r = dk.diagnose(mf.model, mf.params, mf.full_guess())
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
    return 0 if ok else 1


def check_closed_forms() -> bool:
    import numpy as np
    from scipy.io import loadmat
    from dsgelab.steady import SteadyStateError
    print("\nsteady_state_model como estado estacionário")
    ok = True
    f = HERE / "out_import" / "fs2000.mat"
    if f.exists():
        ex = loadmat(f, squeeze_me=True, struct_as_record=False)["ex"]
        ys = dict(zip([str(x).strip() for x in np.atleast_1d(ex.endo_names)], np.atleast_1d(ex.ys)))
        mf = from_mod(HERE / "mods" / "examples" / "fs2000.mod")
        ss = mf.steady_state(analytic=True)
        d = max(abs(ss.values[v] - ys[v]) for v in mf.endogenous)
        print(f"  fs2000: forma fechada contra o e.e. do Dynare  {d:.1e}   resíduo {ss.residual:.1e}")
        ok &= d < 1e-12
    else:
        print("  fs2000: sem out_import/fs2000.mat — rode run_examples.sh")
        ok = False
    try:
        from_mod(HERE / "diagnostics" / "bad_steady.mod").steady_state(analytic=True)
        print("  bad_steady: ACEITO — deveria ser recusado")
        ok = False
    except SteadyStateError as e:
        print(f"  bad_steady: recusado ({str(e)[:90]}...)")
    text = _run_dynare(HERE / "diagnostics" / "bad_steady.mod")
    refused = text is not None and "did not compute the steady state" in text
    print(f"  bad_steady no Dynare: {'recusado' if refused else 'ACEITO'}")
    return ok and refused


def _run_dynare(mod: Path) -> str | None:
    octave = shutil.which("octave") or "/opt/homebrew/bin/octave"
    matlab = Path("/opt/homebrew/opt/dynare/lib/dynare/matlab")
    if not Path(octave).exists() or not matlab.exists():
        return None
    with tempfile.TemporaryDirectory() as d:
        shutil.copy(mod, d)
        out = subprocess.run([octave, "--no-gui", "--eval",
                              f"pkg load statistics datatypes; addpath '{matlab}'; "
                              f"dynare {mod.stem} noclearall nograph;"],
                             cwd=d, capture_output=True, text=True, timeout=600)
    return out.stdout + out.stderr


if __name__ == "__main__":
    sys.exit(main())
