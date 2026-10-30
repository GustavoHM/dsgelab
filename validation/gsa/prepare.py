"""Prepare the stability-mapping cases from Dynare's tests/gsa/ls2003a.mod.

The GPL files are downloaded by fetch.sh and are not versioned here. This script
writes two self-contained drivers into work_gsa/, with the Lubik and Schorfheide
(2003) model and its priors, the include inlined, and without the calibration
blocks, which belong to compare_restrictions.py:

* ``ls2003_prior``: ``sensitivity(stab=1, prior_range=0)``, draws from the prior;
* ``ls2003_range``: ``sensitivity(stab=1, prior_range=1)``, uniform draws over the
  prior's range;
* ``ls2003_explosive``: draws from the prior, with the persistence of technology
  ``rho_A`` uniform on [0.5, 1.1] instead of beta, so that a sixth of the draws
  have an explosive root.
"""
from pathlib import Path
import re

HERE = Path(__file__).resolve().parents[1]
SOURCE = HERE / "mods" / "gsa" / "ls2003a.mod"
WORK = HERE / "work_gsa"
NSAM = 1024


def main():
    text = SOURCE.read_text()
    model = (SOURCE.parent / "ls2003_model.inc").read_text()
    model = model.replace("options_.TeX=1;", "")
    model = re.sub(r"\$[^$]*\$", "", model)                      # TeX names
    priors = re.search(r"estimated_params;.*?\nend;", text, re.S).group(0)
    priors = re.sub(r"/\*.*?\*/", "", priors, flags=re.S)        # the wider priors, commented
    WORK.mkdir(exist_ok=True)
    wide = priors.replace("rho_A ,beta_pdf,0.5,0.2;", "rho_A ,uniform_pdf, , ,0.5,1.1;")
    assert wide != priors
    for name, prior_range, pr in (("ls2003_prior", 0, priors), ("ls2003_range", 1, priors),
                                  ("ls2003_explosive", 0, wide)):
        driver = f"""{model}
{pr}

sensitivity(stab=1, prior_range={prior_range}, nograph, nodisplay, Nsam={NSAM});
"""
        (WORK / f"{name}.mod").write_text(driver)
        print(f"wrote {WORK.name}/{name}.mod")


if __name__ == "__main__":
    main()
