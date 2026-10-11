"""Independent source algebra (Octave) and dynamics (Dynare) checks.

Usage: python validation/heutel/validate.py
Requires Octave and Dynare. Original MATLAB prefixes stop before the absent
AIM dependency; Dynare solves those original coefficient matrices instead.
"""
from pathlib import Path
import json
import hashlib
import os
import re
import subprocess
import sys

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from dsgelab.examples import heutel

HERE = Path(__file__).resolve().parent
OUT = HERE/'out'
OUT.mkdir(exist_ok=True)
DYNARE = os.environ.get('DYNARE_MATLAB', '/opt/homebrew/opt/dynare/lib/dynare/matlab')
OCTAVE = os.environ.get('OCTAVE', '/opt/homebrew/bin/octave')
FILES = dict(planner='SolveSystem.m', tax='SolveSystem_tax.m',
             tax_asym='SolveSystem_tax_AsymInfo.m', quota_asym='SolveSystem_quota_AsymInfo.m')


def quote(path):
    return str(path).replace("'", "''")


def octave_source(variant):
    source = (HERE/'original'/FILES[variant]).read_text()
    prefix = source[:source.index('[b,rts')]
    prefix = re.sub(r"options=optimset\([^;]+;", "options=optimset('Display','off','TolFun',1e-13,'TolX',1e-13,'MaxFunEvals',10000);", prefix)
    if variant == 'planner':
        # Explicit common calibration; the literal source default is tested
        # separately as a nonconvergent/inadmissible calibration in Python.
        prefix = prefix.replace('theta2 = 0.1;', 'theta2 = 2.8;').replace('gamma = 1-.2;', 'gamma = 1-.696;')
    tail = ''
    if variant.endswith('asym'):
        start = source.index('h(11,:)')
        tail = source[start:source.index('[b,rts', start)]
    names = heutel.BASE_NAMES + (() if variant == 'planner' else ('tau','r','lambda','zeta','omega')) + ('a',)
    save = "\nlevels = [" + ','.join(k+'_ss' for k in names) + "];\n"
    save += "save('-mat7-binary','"+quote(OUT/(variant+'.mat'))+"','h','levels');\n"
    program = "addpath('"+quote(HERE/'original')+"');\n"+prefix+tail+save
    if variant == 'tax':
        # Independently evaluate the zero-multiplier coordinate limit from
        # original MATLAB expressions, per unit lambda, without small division.
        matrix_start = prefix.index('h = zeros')
        program += "lambda_ss = 1;\n" + prefix[matrix_start:]
        program += "h_unit = h; save('-mat7-binary','"+quote(OUT/'tax_unit.mat')+"','h_unit');\n"
    script=OUT/(variant+'_source.m'); script.write_text(program)
    with (OUT/(variant+'_source.log')).open('w') as log:
        subprocess.run([OCTAVE,'--no-gui','--quiet',str(script)],stdout=log,stderr=subprocess.STDOUT,check=True)
    return loadmat(OUT/(variant+'.mat'))


def dynare_irfs(variant, names, h):
    n=len(names)
    lines=['var '+' '.join(names)+';', 'varexo eps_a;', 'model(linear);']
    for row in range(n):
        terms=[]
        for j,value in enumerate(h[row]):
            if value == 0: continue
            offset=j//n-1
            var=names[j%n]+(f'({offset:+d})' if offset else '')
            terms.append(f'({value:.17g})*{var}')
        lines.append(' + '.join(terms)+(' - eps_a' if row==0 else '')+' = 0;')
    lines += ['end;', 'shocks; var eps_a; stderr 0.01; end;',
              'steady; check;', 'stoch_simul(order=1,irf=100,nograph,noprint);',
              f"save('-mat7-binary','{variant}_dynare.mat','oo_');"]
    mod=OUT/(variant+'_ref.mod'); mod.write_text('\n'.join(lines)+'\n')
    with (OUT/(variant+'_dynare.log')).open('w') as log:
        subprocess.run([OCTAVE,'--no-gui','--quiet','--eval',
            f"pkg load statistics datatypes; addpath('{quote(DYNARE)}'); dynare {mod.name} noclearall nograph;"],
            cwd=OUT,stdout=log,stderr=subprocess.STDOUT,check=True)
    return loadmat(OUT/(variant+'_dynare.mat'),simplify_cells=True)['oo_']


def main():
    manifest=json.loads((HERE/'source-sha256.json').read_text())
    for name, digest in manifest.items():
        actual=hashlib.sha256((HERE/'original'/name).read_bytes()).hexdigest()
        if actual != digest:
            raise RuntimeError(f"Source checksum changed: {name}")
    results={}; fixtures={}
    for variant in heutel.VARIANTS:
        p=heutel.Params() if variant=='planner' else heutel.Params.source(variant)
        ref=octave_source(variant)
        names,h,ss=heutel.structural_system(variant,p)
        expected_levels=ref['levels'].ravel().copy()
        if variant=='quota_asym': expected_levels[8]=expected_levels[1]
        ss_error=float(np.max(np.abs(np.array([ss[n] for n in names])-expected_levels)))
        matrix_error=float(np.max(np.abs(h-ref['h'])))
        # Preserve raw source h for future tests. Normalize only the solver copy.
        dynare_h=ref['h'].copy()
        if variant=='tax':
            cols=[10,24,38]
            dynare_h[:,cols]=loadmat(OUT/'tax_unit.mat')['h_unit'][:,cols]
        dyn=dynare_irfs(variant,names,dynare_h)
        solution=heutel.solve(variant,p)
        irf=solution.irf('eps_a',size=.01,periods=100,percent=False)
        irf_error=max(float(np.max(np.abs(irf[n]-dyn['irfs'][n+'_eps_a']))) for n in names)
        # Dynare's variance uses sd=.01; compare the same standard deviation.
        moments=solution.moments(cov={'eps_a':.01})
        # Moment API exposes named standard deviations.
        std_ref=np.sqrt(np.maximum(np.diag(dyn['var']),0))
        std_error=max(abs(float(moments['std'][solution.model.position(n)])-float(std_ref[j])) for j,n in enumerate(names))
        results[variant]=dict(steady_max_abs=ss_error,matrix_max_abs=matrix_error,
                              irf_max_abs=irf_error,std_max_abs=std_error)
        fixtures[variant]=dict(names=list(names),levels=expected_levels.tolist(),
                              h=ref['h'].tolist(),irfs={n:np.asarray(dyn['irfs'][n+'_eps_a']).tolist() for n in names},
                              std=std_ref.tolist())
        print(variant,results[variant],flush=True)
        assert ss_error < 1e-7 and matrix_error < 1e-8 and irf_error < 1e-7 and std_error < 1e-7
    (HERE/'reference.json').write_text(json.dumps(fixtures,indent=2)+'\n')
    (HERE/'results.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':
    main()
