"""strip_mod.py: a copy of a .mod that only declares the model, for run_examples.sh.

Dynare's examples follow the model with estimation (fs2000 asks for 20,000 draws and a data
file), graphs and MATLAB code. To compare MODELS, the copy keeps the text up to the end of the
last block that defines the model (model, initval, steady_state_model, shocks), removes any
stand-alone commands in that stretch and appends the same computation for all of them: steady
and an order-1 stoch_simul, 40 periods of IRF and theoretical moments.

Usage:  python strip_mod.py source.mod destination.mod
"""
import re
import sys

BLOCKS = ("model", "initval", "steady_state_model", "shocks")
COMMANDS = r"steady|check|resid|model_diagnostics|write_latex_\w+|stoch_simul"

src, dst = sys.argv[1], sys.argv[2]
text = open(src).read()
masked = re.sub(r"/\*.*?\*/", lambda m: " " * len(m.group(0)), text, flags=re.S)
masked = re.sub(r"(//|%)[^\n]*", lambda m: " " * len(m.group(0)), masked)
last = 0
for m in re.finditer(rf"^\s*({'|'.join(BLOCKS)})\b[^;]*;", masked, flags=re.M):
    end = re.compile(r"^\s*end\s*;", re.M).search(masked, m.end())
    if end:
        last = max(last, end.end())
if not last:
    sys.exit(f"{src}: no model block")
head = text[:last]
head_masked = masked[:last]
for m in reversed(list(re.finditer(rf"^\s*({COMMANDS})\b[^;]*;", head_masked, flags=re.M))):
    head = head[:m.start()] + head[m.end():]
open(dst, "w").write(head + "\n\nsteady;\nstoch_simul(order=1, irf=40, periods=0, nograph, noprint);\n")
