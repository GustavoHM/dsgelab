function export_plots(fname, outfile)
% Gather, after an estimation of fs2000 with mode_check, Metropolis-Hastings and bayesian_irf,
% what compare_plots.py needs: the mode and the bounds mode_check used, its slices, every draw of
% each chain with its log posterior, the draws and responses of the posterior IRFs with their
% statistics, and the numbers drawn by Dynare's own figures (recorded by plots/dyn_saveas.m), after
% two more of them are drawn here: the trace plot and the autocorrelogram of one parameter.
  global M_ estim_params_ options_ bayestopt_ oo_ dataset_ PLOT_CAPTURE
  ex = struct();
  ex.names = bayestopt_.name;
  ex.data = dataset_.data; ex.varobs = options_.varobs;
  M = load([fname '/Output/' fname '_mode.mat']);
  ex.xparam1 = M.xparam1;
  % the bounds of dynare_estimation_init: the prior's quantiles at prior_trunc, inside the bounds
  % of estimated_params
  b = prior_bounds(bayestopt_, options_.prior_trunc);
  lb = [estim_params_.var_exo(:, 3); estim_params_.param_vals(:, 3)];
  ub = [estim_params_.var_exo(:, 4); estim_params_.param_vals(:, 4)];
  ex.lb = max(b.lb, lb); ex.ub = min(b.ub, ub);
  C = load([fname '/graphs/' fname '_check_plot_data.mat']);
  ex.mcheck = C.mcheck;
  % every draw of each chain, burn-in included
  nblck = options_.mh_nblck;
  for blck = 1:nblck
    x = []; lp = [];
    files = dir([fname '/metropolis/' fname '_mh*_blck' int2str(blck) '.mat']);
    for f = 1:numel(files)
      D = load([fname '/metropolis/' fname '_mh' int2str(f) '_blck' int2str(blck) '.mat']);
      x = [x; D.x2]; lp = [lp; D.logpo2];
    end
    ex.draws(:, :, blck) = x; ex.logpo(:, blck) = lp;
  end
  ex.mh_drop = options_.mh_drop;
  % posterior IRFs: the parameters drawn, the responses at each, and Dynare's statistics
  P = load([fname '/metropolis/' fname '_param_irf1.mat']);
  ex.irf_params = P.stock;
  S = load([fname '/metropolis/' fname '_IRF_DSGEs1.mat']);
  ex.irf_stock = S.STOCK_IRF_DSGE;
  ex.irf_vars = options_.varlist;
  ex.exo_names = M_.exo_names;
  ex.irf = oo_.PosteriorIRF.dsge;
  ex.mh_conf_sig = options_.mh_conf_sig;
  % two more figures, for the first deep parameter of the first chain
  pname = M_.param_names{estim_params_.param_vals(1, 1)};
  ex.trace_name = pname;
  trace_plot(options_, M_, estim_params_, 'DeepParameter', 1, pname);
  trace_plot(options_, M_, estim_params_, 'PosteriorDensity', 1);
  mh_autocorrelation_function(options_, M_, estim_params_, 'DeepParameter', 1, pname);
  ex.trace_plot_ma = options_.trace_plot_ma;
  ex.acf_size = options_.mh_autocorrelation_function_size;
  ex.figures = PLOT_CAPTURE;
  save('-mat7-binary', outfile, 'ex');
end
