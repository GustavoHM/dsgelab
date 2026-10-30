"""Prepare the calibration-restriction cases from Dynare's tests/gsa/ls2003a.mod.

The GPL files are downloaded by fetch.sh and are not versioned here. ls2003a.mod
includes the Lubik and Schorfheide (2003) model, declares its priors, and has an
active ``irf_calibration(relative_irf)`` block, a commented ``irf_calibration``
block in absolute units and a ``moment_calibration`` block with two commented
lines. This script writes two self-contained drivers into work_restrictions/:

* ``ls2003_rel``: the active relative_irf block;
* ``ls2003_abs``: the commented block, in absolute units.

Both carry the moment block with its commented lines restored (the variance and
the -4:4 cross-correlations, a range), the include inlined and the ``@#for``
loops expanded, since dsgelab's importer reads neither. A stoch_simul sets up the
decision rule, a verbatim block saves Dynare's restrictions at the calibration,
and sensitivity(prior_range=0) evaluates them over draws from the prior.
"""
from pathlib import Path
import re

HERE = Path(__file__).resolve().parents[1]
SOURCE = HERE / "mods" / "gsa" / "ls2003a.mod"
WORK = HERE / "work_restrictions"
NSAM = 512


def expand_for(text: str) -> str:
    """``@#for i in a:b`` ... ``@#endfor`` with ``@{i}``, one copy per value."""
    pattern = re.compile(r"@#for\s+(\w+)\s+in\s+(-?\d+)\s*:\s*(-?\d+)\s*\n(.*?)@#endfor", re.S)

    def repl(m):
        name, a, b, body = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
        return "".join(body.replace("@{" + name + "}", str(i)) for i in range(a, b + 1))
    return pattern.sub(repl, text)


def main():
    text = SOURCE.read_text()
    model = (SOURCE.parent / "ls2003_model.inc").read_text()
    model = model.replace("options_.TeX=1;", "")
    model = re.sub(r"\$[^$]*\$", "", model)                      # TeX names
    priors = re.search(r"estimated_params;.*?\nend;", text, re.S).group(0)
    priors = re.sub(r"/\*.*?\*/", "", priors, flags=re.S)        # the wider priors, commented
    rel = re.search(r"irf_calibration\(relative_irf\);.*?\nend;", text, re.S).group(0)
    absolute = re.search(r"/\*\s*(irf_calibration;.*?\nend;)\s*\*/", text, re.S).group(1)
    moments = re.search(r"moment_calibration;.*?\nend;", text, re.S).group(0)
    moments = re.sub(r"^//(y_obs,)", r"\1", moments, flags=re.M)
    WORK.mkdir(exist_ok=True)
    for name, irf in (("ls2003_rel", rel), ("ls2003_abs", absolute)):
        irf = "\n".join(line for line in irf.splitlines() if not line.lstrip().startswith("//"))
        driver = f"""{model}
{priors}

{expand_for(irf)}

{expand_for(moments)}

stoch_simul(order=1, irf=0, nograph, noprint, nomoments, nocorr, nodecomposition);
verbatim;
restrictions_point(M_, options_, oo_, '{name}_point.mat');
end;
sensitivity(prior_range=0, nograph, nodisplay, Nsam={NSAM});
"""
        (WORK / f"{name}.mod").write_text(driver)
        print(f"wrote {WORK.name}/{name}.mod")


if __name__ == "__main__":
    main()
