"""strip_mod.py — a copy of a .mod that only declares the model, for run_examples.sh.

Os exemplos do Dynare seguem o modelo com estimação (fs2000 pede 20 000 sorteios e um arquivo de
dados), gráficos e código MATLAB. Para comparar MODELOS, a cópia guarda o texto até o fim do último
bloco que define o modelo — model, initval, steady_state_model, shocks —, tira os comandos soltos
que houver nesse trecho e acrescenta o mesmo cálculo para todos: steady e stoch_simul de ordem 1,
40 períodos de IRF e momentos teóricos.

Uso:  python strip_mod.py origem.mod destino.mod
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
