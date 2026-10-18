"""working_copy.py: a .mod cut at its first identification or estimation command.

Dynare's test files go on after the identification command with checks, plots and LaTeX, and
fs2000 ends with an estimation that needs its data file. The copy keeps the text before the
first such command and appends one identification command with the given options.

Usage:  python working_copy.py source.mod destination.mod "advanced=1,max_dim_cova_group=3"
"""
import re
import sys


def working_copy(text: str, options: str) -> str:
    masked = re.sub(r"/\*.*?\*/", lambda m: " " * len(m.group(0)), text, flags=re.S)
    masked = re.sub(r"(//|%)[^\n]*", lambda m: " " * len(m.group(0)), masked)
    m = re.search(r"^\s*(identification|estimation)\s*[(;]", masked, flags=re.M)
    if not m:
        raise ValueError("no identification or estimation command")
    return text[:m.start()].rstrip() + f"\n\nidentification({options});\n"


if __name__ == "__main__":
    src, dst, options = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(src) as f:
        out = working_copy(f.read(), options)
    with open(dst, "w") as f:
        f.write(out)
