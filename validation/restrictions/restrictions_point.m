function restrictions_point(M_, options_, oo_, outfile)
% Dynare's endogenous_prior_restrictions at the calibrated parameters.
  if isempty(options_.qz_criterium)
    options_.qz_criterium = 1 + 1e-6;
  end
  [T, R, ~, info, dr] = dynare_resolve(M_, options_, oo_.dr, oo_.steady_state, ...
                                       oo_.exo_steady_state, oo_.exo_det_steady_state);
  if info(1)
    error('no solution at the calibration');
  end
  [info, info_irf, info_moment, data_irf, data_moment] = endogenous_prior_restrictions( ...
      T, R, M_, options_, dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
  pt = struct();
  pt.info = info;
  pt.info_irf = info_irf;
  pt.info_moment = info_moment;
  pt.data_irf = data_irf;
  pt.data_moment = data_moment;
  pt.params = M_.params;
  pt.param_names = M_.param_names;
  pt.Sigma_e = M_.Sigma_e;
  save('-mat7-binary', outfile, 'pt');
end
