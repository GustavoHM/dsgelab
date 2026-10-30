function export_mom(fname, outfile)
  S = load([fname '/method_of_moments/' fname '_mom_results.mat']);
  oo = S.oo_; o = S.options_mom_; M = S.M_;
  ex = struct();
  ex.mom = oo.mom;
  if isfield(ex.mom, 'irf_model_varobs'), ex.mom = rmfield(ex.mom, 'irf_model_varobs'); end
  ex.nobs = o.nobs;
  ex.simulation_multiple = 1;
  if isfield(o.mom, 'simulation_multiple'), ex.simulation_multiple = o.mom.simulation_multiple; end
  ex.param_names = M.param_names; ex.params = M.params;
  ex.exo_names = M.exo_names; ex.Sigma_e = M.Sigma_e;
  if isfield(o.mom, 'shock_series'), ex.shock_series = o.mom.shock_series; end
  ex.matched_moments = M.matched_moments; ex.endo_names = M.endo_names;
  save('-mat7-binary', outfile, 'ex');
end
