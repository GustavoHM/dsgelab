function export_irf(fname, outfile)
  S = load([fname '/method_of_moments/' fname '_mom_results.mat']);
  D = load([fname '/method_of_moments/' fname '_mode.mat']);
  ex = struct();
  ex.mom = S.oo_.mom;
  ex.options = S.options_mom_;
  ex.hessian = D.hh;
  ex.mode = D.xparam1;
  ex.fval = D.fval;
  ex.parameter_names = D.parameter_names;
  ex.params = S.M_.params;
  ex.param_names = S.M_.param_names;
  ex.Sigma_e = S.M_.Sigma_e;
  ex.exo_names = S.M_.exo_names;
  save('-mat7-binary', outfile, 'ex');
end
