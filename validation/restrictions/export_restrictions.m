function export_restrictions(fname, outfile)
% Gather the sensitivity toolbox's prior draws and restriction values for compare_restrictions.py.
  global M_ estim_params_ options_
  G = load([fname '/gsa/' fname '_prior_restrictions']);
  P = load([fname '/gsa/' fname '_prior.mat']);
  load([fname '_point.mat'], 'pt');
  ex = struct();
  ex.point = pt;
  ex.lpmat = [P.lpmat0 P.lpmat];               % every draw, stderr first, as GSA orders them
  ex.irestrictions = G.irestrictions;           % the draws Dynare could solve
  ex.mat_irf = G.mat_irf;
  ex.mat_moment = G.mat_moment;
  ex.indx_irf = G.indx_irf;
  ex.indx_moment = G.indx_moment;
  ex.names = [M_.exo_names(estim_params_.var_exo(:, 1)); M_.param_names(estim_params_.param_vals(:, 1))];
  ex.relative_irf = options_.relative_irf;
  ex.irf = G.endo_prior_restrictions.irf;
  ex.moment = G.endo_prior_restrictions.moment;
  save('-mat7-binary', outfile, 'ex');
end
