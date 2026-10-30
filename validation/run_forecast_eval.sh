#!/usr/bin/env bash
# run_forecast_eval.sh: references for the out-of-sample forecast evaluation.
#
# Usage:  bash run_forecast_eval.sh [recursive|dm ...]   (no arguments: both)
#
#   recursive  Dynare's recursive estimation of fs2000.mod (Schorfheide, 2000), with model, priors
#              and data as they are: estimation(nobs=[176:184], mode_compute=4, mh_replic=0,
#              forecast=8). For each of the nine samples, the posterior mode, the posterior
#              kernel at the mode, and the 8-period forecast of the observables with 90% bands.
#              Dynare starts the optimisation of every sample after the first from the first
#              sample's mode. options_.riccati_tol = 1e-16 as in run_estimation.sh.
#   dm         R's forecast::dm.test (Diebold-Mariano with the Harvey-Leybourne-Newbold
#              correction) on two autocorrelated error series drawn in R with a fixed seed, for
#              horizons 1 to 4, losses |e| and e^2, both variance estimators and the three
#              alternatives.
#
# Exports out_forecast_eval/fs2000_recursive.mat (with the full sample of 192 observations), out_forecast_eval/dm_errors.csv and
# out_forecast_eval/dm_tests.csv.
set -euo pipefail
cd "$(dirname "$0")"

OCTAVE="${OCTAVE:-/opt/homebrew/bin/octave}"
RSCRIPT="${RSCRIPT:-Rscript}"
VARIANTS=("$@")
[ ${#VARIANTS[@]} -eq 0 ] && VARIANTS=(recursive dm)
mkdir -p out_forecast_eval

for v in "${VARIANTS[@]}"; do
  if [ "$v" = dm ]; then
    echo ">>> dm"
    "$RSCRIPT" - <<'R' > out_forecast_eval/dm.log 2>&1 || { echo "FAILED dm"; tail -20 out_forecast_eval/dm.log; exit 1; }
suppressMessages(library(forecast))
set.seed(20261030)
n <- 60
e1 <- as.numeric(arima.sim(list(ma = c(0.6, 0.3)), n))
e2 <- as.numeric(0.9 * arima.sim(list(ma = 0.5), n) + 0.2)
write.csv(data.frame(e1 = e1, e2 = e2), "out_forecast_eval/dm_errors.csv", row.names = FALSE)
rows <- list()
for (h in 1:4) for (power in 1:2) for (ve in c("acf", "bartlett")) for (alt in c("two.sided", "less", "greater")) {
  r <- withCallingHandlers(dm.test(e1, e2, alternative = alt, h = h, power = power, varestimator = ve),
                           warning = function(w) invokeRestart("muffleWarning"))
  rows[[length(rows) + 1]] <- data.frame(h = h, power = power, varestimator = ve, alternative = alt,
                                         statistic = unname(r$statistic), p_value = r$p.value,
                                         h_used = unname(r$parameter[1]))
}
out <- do.call(rbind, rows)
write.csv(out, "out_forecast_eval/dm_tests.csv", row.names = FALSE)
cat(sprintf("forecast %s, %d tests\n", as.character(packageVersion("forecast")), nrow(out)))
cat("saved out_forecast_eval/dm_tests.csv\n")
R
    tail -1 out_forecast_eval/dm.log
    continue
  fi
  if [ "$v" != recursive ]; then echo "unknown variant $v"; exit 1; fi
  if [ -z "${DYNARE_MATLAB:-}" ]; then
    for c in /opt/homebrew/opt/dynare/lib/dynare/matlab /opt/homebrew/lib/dynare/matlab \
             /usr/local/opt/dynare/lib/dynare/matlab /usr/lib/dynare/matlab \
             /Applications/Dynare/*/matlab; do
      [ -d "$c" ] && DYNARE_MATLAB="$c" && break
    done
  fi
  [ -n "${DYNARE_MATLAB:-}" ] || { echo "FATAL: Dynare's matlab/ directory not found. Set DYNARE_MATLAB."; exit 1; }
  echo ">>> fs2000 — recursive"
  W=work_forecast_eval; rm -rf "$W"; mkdir -p "$W"
  python3 - mods/examples/fs2000.mod "$W/fs2000.mod" <<'PY'
import re, sys
src, dst = sys.argv[1], sys.argv[2]
line = ("options_.riccati_tol = 1e-16;\n"
        "estimation(order=1, datafile=fs2000_data, loglinear, logdata, nograph, nodisplay, "
        "plot_priors=0, nobs=[176:184], mode_compute=4, mh_replic=0, forecast=8) gp_obs gy_obs;")
text = re.sub(r"^estimation\(.*$", lambda m: line, open(src).read(), flags=re.M)
open(dst, "w").write(text)
PY
  cp mods/examples/fs2000_data.m "$W/"
  rm -f out_forecast_eval/fs2000_recursive.mat
  ( cd "$W" && "$OCTAVE" --no-gui --eval "
    pkg load statistics datatypes;
    addpath '$DYNARE_MATLAB';
    dynare fs2000 noclearall nograph;
    ex = struct();
    ex.estim_names = bayestopt_.name; ex.varobs = options_.varobs;
    fs2000_data;   % the full sample: dataset_ holds only the last estimation window
    ex.data = [gp_obs(:), gy_obs(:)];
    ex.nobs = 176:184;
    ex.mode = []; ex.log_density = [];
    vars = {'gp_obs', 'gy_obs'};
    for i = 1:numel(ex.nobs)
      o = oo_recursive_{ex.nobs(i)};
      ex.mode(:, i) = o.posterior.optimization.mode(:);
      ex.log_density(i) = o.posterior.optimization.log_density;
      for k = 1:numel(vars)
        ex.mean.(vars{k})(:, i) = o.forecast.Mean.(vars{k})(:);
        ex.lower.(vars{k})(:, i) = o.forecast.HPDinf.(vars{k})(:);
        ex.upper.(vars{k})(:, i) = o.forecast.HPDsup.(vars{k})(:);
      end
    end
    save('-mat7-binary', '../out_forecast_eval/fs2000_recursive.mat', 'ex');
    printf('saved out_forecast_eval/fs2000_recursive.mat\n');
  " > ../out_forecast_eval/fs2000_recursive.log 2>&1 ) \
    || { echo "FAILED recursive, see out_forecast_eval/fs2000_recursive.log"; tail -20 out_forecast_eval/fs2000_recursive.log; exit 1; }
  grep '^saved' out_forecast_eval/fs2000_recursive.log
  rm -rf "$W"
done
echo "done."
