"""Prepare the mixed-frequency case from Dynare's examples/fs2000.mod.

Output growth is observed only as its four-quarter mean, once a year, and inflation every
quarter. Dynare has no aggregation option, so the driver does what a user of Dynare does: an
auxiliary variable gy_obs_a = (gy_obs + gy_obs(-1) + gy_obs(-2) + gy_obs(-3))/4 with its steady
state, observed in place of gy_obs, and data with NaN in the three quarters between observations
(fs2000_mf_data.m, which runs Dynare's fs2000_data.m and adds the series). With loglinear, the
log-linearised aggregate is the mean of the log deviations, which is dsgelab's "mean" of a log
observable. The estimation line evaluates the likelihood at two points and smooths at the first,
in the work_mixed/fs2000_mf_point variant, and finds the posterior mode in fs2000_mf_mode.
"""
from pathlib import Path
import re

HERE = Path(__file__).resolve().parents[1]
SOURCE = HERE / "mods" / "examples" / "fs2000.mod"
WORK = HERE / "work_mixed"

DATA = """fs2000_data;
gy_obs_a = NaN(size(gy_obs));
for t = 4:4:length(gy_obs)
  gy_obs_a(t) = mean(gy_obs(t-3:t));
end
"""

COMMON = "order=1, datafile=fs2000_mf_data, loglinear, logdata, nograph, nodisplay, plot_priors=0"
LINES = {
    "fs2000_mf_point": f"estimation({COMMON}, mode_compute=0, mh_replic=0, smoother) "
                       f"gy_obs gp_obs y c gy_obs_a;",
    "fs2000_mf_mode": f"estimation({COMMON}, mode_compute=4, mh_replic=0);",
}


def main():
    text = SOURCE.read_text()
    text = re.sub(r"^(var\s+m\b)", r"var gy_obs_a;\n\1", text, count=1, flags=re.M)
    text = text.replace("gy_obs = dA*y/y(-1);",
                        "gy_obs = dA*y/y(-1);\n"
                        "[name='Annual mean of GDP growth']\n"
                        "gy_obs_a = (gy_obs + gy_obs(-1) + gy_obs(-2) + gy_obs(-3))/4;")
    text = text.replace("  gy_obs = dA;\nend;", "  gy_obs = dA;\n  gy_obs_a = dA;\nend;")
    text = text.replace("varobs gp_obs gy_obs;", "varobs gp_obs gy_obs_a;")
    assert text.count("gy_obs_a") == 4
    WORK.mkdir(exist_ok=True)
    (WORK / "fs2000_mf_data.m").write_text(DATA)
    for name, line in LINES.items():
        out = re.sub(r"^estimation\(.*$", lambda m: "options_.riccati_tol = 1e-16;\n" + line,
                     text, flags=re.M)
        (WORK / f"{name}.mod").write_text(out)
        print(f"wrote {WORK.name}/{name}.mod")


if __name__ == "__main__":
    main()
