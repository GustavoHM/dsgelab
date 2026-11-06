function export_decomposition(outfile, varlist, cf_replic, cf_periods)
% Gather, after the shock decompositions and the conditional forecast of fs2000 at the prior
% mean (run_plots.sh decomposition), what compare_plots.py needs: the data and the point, the
% initial condition decomposition, the real-time pools (realtime, one-step conditional and
% forecast) and the vintages saved, the decompositions that plot_shock_decomposition draws with
% type=yoy and type=aoa and with realtime=1, 2 and 3, rolling and at a vintage, and the
% conditional forecast with its bands and the normal draws it used, drawn again from the state
% of randn recorded just before the conditional_forecast command.
  global M_ options_ oo_ estim_params_ bayestopt_ dataset_ CF_RNG
  ex = struct();
  ex.data = dataset_.data; ex.varobs = options_.varobs;
  ex.estim_names = bayestopt_.name;
  ex.endo_names = M_.endo_names; ex.exo_names = M_.exo_names;
  x = [];
  for i = 1:size(estim_params_.var_exo, 1), j = estim_params_.var_exo(i,1); x(end+1,1) = sqrt(M_.Sigma_e(j,j)); end
  for i = 1:size(estim_params_.param_vals, 1), x(end+1,1) = M_.params(estim_params_.param_vals(i,1)); end
  ex.point = x;
  ex.varlist = varlist;
  ex.initval = oo_.initval_decomposition;
  ex.shock_decomposition = oo_.shock_decomposition;
  ex.realtime_pool = oo_.realtime_shock_decomposition.pool;
  ex.conditional_pool = oo_.conditional_shock_decomposition.pool;
  ex.forecast_pool = oo_.realtime_forecast_shock_decomposition.pool;
  ex.realtime = oo_.realtime_shock_decomposition;
  ex.realtime_forecast = oo_.realtime_forecast_shock_decomposition;
  ex.realtime_conditional = oo_.realtime_conditional_shock_decomposition;
  ex.conditional = oo_.conditional_shock_decomposition;
  ex.forecast = options_.shock_decomp.forecast;
  ex.save_realtime = options_.shock_decomp.save_realtime;
  % what plot_shock_decomposition draws, without drawing it (its get_decomp_only)
  opts = options_;
  opts.plot_shock_decomp.use_shock_groups = '';
  ex.vintage = opts.shock_decomp.save_realtime(2);
  types = {'yoy', 'aoa'};
  for k = 1:2
    o = opts; o.plot_shock_decomp.type = types{k}; o.plot_shock_decomp.realtime = 0;
    ex.(['type_' types{k}]) = plot_shock_decomposition(M_, oo_, o, varlist, true);
  end
  for r = 1:3
    o = opts; o.plot_shock_decomp.type = ''; o.plot_shock_decomp.realtime = r;
    o.plot_shock_decomp.vintage = ex.vintage;
    ex.(['realtime_' int2str(r)]) = plot_shock_decomposition(M_, oo_, o, varlist, true);
    o.plot_shock_decomp.vintage = 0;
    ex.(['realtime_' int2str(r) '_pool']) = plot_shock_decomposition(M_, oo_, o, varlist, true);
  end
  % the conditional forecast, its bands, and the draws behind them
  ex.cf = oo_.conditional_forecast;
  ex.cf_replic = cf_replic;
  ex.cf_periods = cf_periods;
  randn('state', CF_RNG);
  Z1 = zeros(M_.exo_nbr, cf_periods, cf_replic); Z2 = Z1;
  for b = 1:cf_replic, Z1(:, :, b) = randn(M_.exo_nbr, cf_periods); end
  for b = 1:cf_replic, Z2(:, :, b) = randn(M_.exo_nbr, cf_periods); end
  ex.cf_normals = Z1; ex.cf_normals_uncond = Z2;
  save('-mat7-binary', outfile, 'ex');
end
