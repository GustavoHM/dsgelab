function dyn_saveas(h, fname, nodisplay, graph_format)
% Stands in for Dynare's dyn_saveas during run_plots.sh: instead of writing the figure to a file,
% it records what each panel draws, so that compare_plots.py can compare the numbers of Dynare's
% own figures. For each figure: its file name, and for each panel its title and the x and y data
% of every line and bar series, in the order they were drawn.
  global PLOT_CAPTURE
  if isempty(PLOT_CAPTURE)
    PLOT_CAPTURE = {};
  end
  [~, base] = fileparts(fname);
  panels = {};
  ax = flipud(findobj(h, 'type', 'axes'));
  for a = ax(:)'
    p = struct();
    p.title = get(get(a, 'title'), 'string');
    p.x = {};
    p.y = {};
    p.type = {};
    children = flipud(get(a, 'children'));
    for c = children(:)'
      t = get(c, 'type');
      if any(strcmp(t, {'line', 'hggroup'}))
        p.x{end + 1} = get(c, 'xdata');
        p.y{end + 1} = get(c, 'ydata');
        p.type{end + 1} = t;
      end
    end
    panels{end + 1} = p;
  end
  PLOT_CAPTURE{end + 1} = struct('name', base, 'panels', {panels});
  close(h);
end
