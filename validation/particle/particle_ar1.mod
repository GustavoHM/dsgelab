// A scalar AR(1) observed without transformation, for the particle filter
// (run_particle.sh, which rewrites the stoch_simul line for each order) and the
// inversion filter (run_inversion.sh). The floor at -100 never binds: it is there
// because Dynare's occbin.IVF_core needs an occbin_constraints block. Written for
// dsgelab's validation; MIT, like the package.
var y;
varexo e;
parameters rho;
rho = 0.5;
model;
[name='floor', bind='floor']
y = -100;
[name='floor', relax='floor']
y = rho*y(-1) + e;
end;
steady_state_model;
y = 0;
end;
shocks;
var e = 0.04;
end;
varobs y;
occbin_constraints;
name 'floor'; bind y < -100; relax y > -100;
end;
stoch_simul(order=1, irf=0, periods=0);
