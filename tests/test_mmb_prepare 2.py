from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from validation.mmb.prepare import _assignments, prepare_text


def test_assignment_range_accepts_only_scalar_assignments(tmp_path: Path):
    source = tmp_path / "helper.m"
    source.write_text("a = 1;\n% comment\nb = a^2;\nstop\n")
    assert _assignments(source, "a = 1;", "stop") == "a = 1;\nb = a^2;"
    source.write_text("a = 1;\nfor i = 1:2\nend\nstop\n")
    with pytest.raises(ValueError, match="unsupported MATLAB"):
        _assignments(source, "a = 1;", "stop")


def test_gnss_values_come_from_the_text_file(tmp_path: Path):
    np.savetxt(tmp_path / "median_values.txt", np.arange(1, 26) / 100)
    (tmp_path / "EA_GNSS10.mod").write_text("""
load median_values.txt;
coeffs = median_values;
parameters eps_bh eps_be eps_b rho;
eps_bh = 2;
eps_be = 4;
eps_b = mean([eps_bh,eps_be]);
rho = coeffs(2);
""")
    text = prepare_text("EA_GNSS10", tmp_path)
    assert "rho = 0.02;" in text
    assert "eps_b = (eps_bh+eps_be)/2;" in text
    assert "coeffs(" not in text
    assert "load median_values" not in text


def test_aino_values_come_from_the_mat_file(tmp_path: Path):
    savemat(tmp_path / "all_parameters_February2016.mat",
            {"alpha": 0.125, "beta": 0.25, "seepsX": 0.5})
    (tmp_path / "FI_AINO16.mod").write_text("""
% parameters are loaded from MATLAB
parameters alpha beta;
load all_parameters_February2016;
seepsX=pp.seepsX;
model(linear);
""")
    text = prepare_text("FI_AINO16", tmp_path)
    assert "alpha = 0.125;" in text
    assert "beta = 0.25;" in text
    assert "seepsX = 0.5;" in text
    assert "pp." not in text
    assert "load all_parameters" not in text


def test_aino_refuses_an_incomplete_mat_file(tmp_path: Path):
    savemat(tmp_path / "all_parameters_February2016.mat", {"alpha": 0.125})
    (tmp_path / "FI_AINO16.mod").write_text("""
parameters alpha beta;
load all_parameters_February2016;
model(linear);
""")
    with pytest.raises(ValueError, match="beta"):
        prepare_text("FI_AINO16", tmp_path)
