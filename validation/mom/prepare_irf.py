"""Prepare the frequentist first example of Dynare's rbc_irf_matching.mod."""
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SOURCE = HERE / "mods" / "method_of_moments" / "rbc_irf_matching.mod"
WORK = HERE / "work_mom"


def main():
    text = SOURCE.read_text()
    prefix = text[:text.index("method_of_moments(")]
    WORK.mkdir(exist_ok=True)
    for suffix, optimizer in (("0", "0"), ("5", "5, additional_optimizer_steps=[4]")):
        driver = prefix + (
            f"method_of_moments(mom_method=irf_matching, order=1, mode_compute={optimizer}, "
            "silent_optimizer, nograph);\n"
        )
        (WORK / f"rbc_irf_{suffix}.mod").write_text(driver)


if __name__ == "__main__":
    main()
