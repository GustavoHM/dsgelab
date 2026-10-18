function export_ident(fname, out)
% export_ident(fname, out): what Dynare's identification found at the prior mean, saved to out.
%   Each criterion's verdict (rank, sensitive parameters, dependent sets, weak parameters,
%   multicollinearity), the moments and their Jacobian, Qu and Tkachenko's G, the size of
%   Komunjer and Ng's matrix, and the brute-force collinearity search. Read by
%   compare_identification.py.
  global M_ options_ estim_params_
  d = dir([fname '/identification/*_prior_mean_identif.mat']);
  S = load([fname '/identification/' d(1).name]);
  ex = struct();
  ex.param_names = M_.param_names;
  ex.exo_names = M_.exo_names;
  ex.varobs = options_.varobs;
  % Dynare's column order: shock standard deviations, then the model's parameters
  ex.stderr_index = estim_params_.var_exo(:, 1);
  ex.param_index = estim_params_.param_vals(:, 1);
  tests = {'moments', 'spectrum', 'minimal'};
  structs = {S.ide_moments_point, S.ide_spectrum_point, S.ide_minimal_point};
  fields = {'cond', 'rank', 'ind0', 'indno', 'jweak', 'jweak_pair', 'Mco'};
  for t = 1:3
    for f = 1:numel(fields)
      if isfield(structs{t}, fields{f})
        ex.([tests{t} '_' fields{f}]) = structs{t}.(fields{f});
      end
    end
  end
  ex.MOMENTS = S.ide_moments_point.MOMENTS;
  ex.dMOMENTS = S.ide_moments_point.dMOMENTS;
  ex.dSPECTRUM = S.ide_spectrum_point.dSPECTRUM;
  ex.dMINIMAL_size = size(S.ide_minimal_point.dMINIMAL);
  if isfield(S.ide_moments_point, 'cosndMOMENTS')
    ex.cosn = S.ide_moments_point.cosndMOMENTS;
    pars = S.ide_moments_point.pars;
    P = NaN(size(pars, 1), size(pars, 2), size(pars, 2));
    for i = 1:size(pars, 1)
      for l = 1:size(pars, 2)
        P(i, l, 1:numel(pars{i, l})) = pars{i, l};
      end
    end
    ex.pars = P;
  end
  ex.tol_rank = S.store_options_ident.tol_rank;
  save('-mat7-binary', out, '-struct', 'ex');
end
