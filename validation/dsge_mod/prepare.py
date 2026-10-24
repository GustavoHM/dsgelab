"""Prepare source-specific working copies without changing model equations or calibration.

Aguiar–Gopinath interleaves MATLAB plotting code with alternative shock calibrations. Keep the
model through steady_state_model and select the Mexico/Table 4 shock block, as in Dynare's default
macro branch. Hansen's macro-processor title assignments intentionally have no semicolon; add only
those statement terminators so the lightweight from_mod statement reader sees the following var
block separately. The downloaded originals remain unchanged.
"""
import re
import sys
from pathlib import Path

src, dst = map(Path, sys.argv[1:])
text = src.read_text()

if src.stem == "Aguiar_Gopinath_2007":
    match = re.search(r"^steady_state_model\s*;", text, flags=re.M)
    if match is None:
        sys.exit(f"{src}: no steady_state_model block")
    end = re.search(r"^end\s*;", text[match.end():], flags=re.M)
    if end is None:
        sys.exit(f"{src}: unterminated steady_state_model block")
    text = text[:match.end() + end.end()] + """
shocks;
var eps_g; stderr 2.81/100;
var eps_z; stderr 0.48/100;
end;
"""
elif src.stem == "Hansen_1985":
    text = re.sub(r"(title_string\s*=\s*'[^']*')", r"\1;", text)

dst.write_text(text)
