function export_gsa(fname, outfile)
% Gather the stability mapping of the sensitivity toolbox for compare_gsa.py: the draws, their
% classification, and the Smirnov tests and correlations of Dynare's Monte Carlo filtering, run
% here on the same groups that stability_mapping.m forms.
  global M_ estim_params_ options_
  P = load([fname '/gsa/' fname '_prior.mat']);
  ex = struct();
  ex.lpmat = [P.lpmat0 P.lpmat];                 % every draw, stderr first, as GSA orders them
  ex.nshock = size(P.lpmat0, 2);
  ex.names = [M_.exo_names(estim_params_.var_exo(:, 1)); M_.param_names(estim_params_.param_vals(:, 1))];
  ex.istable = P.istable; ex.iindeterm = P.iindeterm; ex.ixun = P.ixun; ex.iwrong = P.iwrong;
  ex.infox = P.infox;
  ex.egg = P.egg; ex.nspred = P.nspred; ex.nboth = P.nboth; ex.nfwrd = P.nfwrd;
  ex.qz_criterium = options_.qz_criterium;
  N = size(P.lpmat, 1); itot = 1:N;
  isolve = itot(~ismember(itot, P.iwrong));
  groups = {'stable', P.istable, itot(~ismember(itot, P.istable));
            'indeterminate', isolve(~ismember(isolve, P.iindeterm)), P.iindeterm;
            'explosive', isolve(~ismember(isolve, P.ixun)), P.ixun};
  for g = 1:size(groups, 1)
    ib = groups{g, 2}; inb = groups{g, 3};
    s = struct(); s.ib = ib; s.inb = inb;
    if ~isempty(ib) && ~isempty(inb)
      % the loop of gsa.stability_mapping_univariate, without its plots
      for j = 1:size(P.lpmat, 2)
        [~, s.proba(j, 1), s.dproba(j, 1)] = gsa.smirnov_test(P.lpmat(ib, j), P.lpmat(inb, j));
      end
      if length(ib) > 10 && length(inb) > 10
        [s.c_beha, s.p_beha] = corrcoef(P.lpmat(ib, :));
        [s.c_nobeha, s.p_nobeha] = corrcoef(P.lpmat(inb, :));
      end
    end
    ex.(groups{g, 1}) = s;
  end
  save('-mat7-binary', outfile, 'ex');
end
