function export_identification(fname, outfile)
% Gather, after Dynare's identification of fs2000 at the prior mean with advanced=1, what
% compare_plots.py needs: the information matrix and the strength and sensitivity measures
% Dynare computed from it, the sensitivity of the moments, the parameters, the simulated sample
% the information matrix was computed on, and the numbers drawn by Dynare's own figures
% (recorded by plots/dyn_saveas.m). The sample is the one Dynare's dsge_likelihood received for
% the asymptotic Hessian, kept by the copy of it that run_plots.sh puts first on the path.
  global M_ options_ estim_params_ PLOT_CAPTURE IDENT_SAMPLE
  d = dir([fname '/identification/*_prior_mean_identif.mat']);
  S = load([fname '/identification/' d(1).name]);
  ex = struct();
  % Dynare's column order: shock standard deviations, then the model's parameters
  ex.exo_names = M_.exo_names; ex.param_names = M_.param_names;
  ex.stderr_index = estim_params_.var_exo(:, 1); ex.param_index = estim_params_.param_vals(:, 1);
  ex.params = S.ide_hess_point.params;
  ex.hess = S.ide_hess_point;
  ex.si_dMOMENTSnorm = S.ide_moments_point.si_dMOMENTSnorm;
  ex.periods = S.store_options_ident.periods;
  ex.data = IDENT_SAMPLE;
  ex.varobs = options_.varobs;
  ex.figures = PLOT_CAPTURE;
  save('-mat7-binary', outfile, 'ex');
end
