"""Prepare first-order An--Schorfheide GMM/SMM comparisons from Dynare's test file.

The GPL model and data are downloaded by fetch.sh and are not versioned here. This
script keeps the model through estimated_params, replaces the larger moment list
and method_of_moments command, and writes the working cases used by run_mom.sh and
compare_mom.py.
"""
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parents[1]
SOURCE = HERE / "mods" / "method_of_moments" / "AnScho_MoM_common.inc"
WORK = HERE / "work_mom"

MOMENTS = """matched_moments;
INT; YGR; INFL;
INT*INT; YGR*YGR; INFL*INFL; INT*YGR; INT*INFL; YGR*INFL;
INT*INT(-1); YGR*YGR(-1); INFL*INFL(-1);
end;
"""

FIVE_PARAMS = """estimated_params;
  rhor,            0.50,         1e-5,        0.99999;
  gamst,           0.75,         -5,          5;
  stderr e_r,      0.30,         1e-8,        5;
  stderr e_g,      0.30,         1e-8,        5;
  stderr e_z,      0.30,         1e-8,        5;
end;
"""


def main():
    text = SOURCE.read_text()
    model = text[:text.index("matched_moments;")]
    WORK.mkdir(exist_ok=True)
    (WORK / "AnScho_model.inc").write_text(model)
    shutil.copyfile(SOURCE.with_name("AnScho_MoM_data_2.mat"), WORK / "AnScho_MoM_data_2.mat")

    simulation = """@#define estimParams = 1
@#include "AnScho_model.inc"
set_dynare_seed(20261024);
stoch_simul(order=1, periods=750, drop=500, irf=0, nomoments, nocorr,
            nodecomposition, nograph, noprint);
send_endogenous_variables_to_workspace;
save('AnScho_MoM_data_1.mat', options_.varobs{:});
"""
    (WORK / "anscho_sim1.mod").write_text(simulation)

    for method in ("GMM", "SMM"):
        for suffix, optimizer, data, five in (
            ("0", "0", 2, False), ("4", "4", 2, False),
            ("d1", "4", 1, False), ("e5", "4", 1, True),
        ):
            name = f"anscho_{method}_{suffix}"
            params = FIVE_PARAMS if five else ""
            extra = ", optim = ('TolFun', 1e-8, 'MaxIter', 3000)" if five else ""
            definition = 99 if five else 1
            driver = f"""@#define estimParams = {definition}
@#include "AnScho_model.inc"
{params}{MOMENTS}method_of_moments(mom_method = {method}, datafile = 'AnScho_MoM_data_{data}.mat', order = 1,
                  weighting_matrix = ['DIAGONAL', 'OPTIMAL'], mode_compute = {optimizer}, se_tolx = 1e-5{extra},
                  silent_optimizer, nograph);
"""
            (WORK / f"{name}.mod").write_text(driver)


if __name__ == "__main__":
    main()
