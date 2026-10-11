// A real business cycle model with a distortionary labour tax as the planner's instrument,
// rebated lump sum. Written for dsgelab's validation; MIT, like the package.
//
// The competitive equilibrium with tau = 0 is efficient, so the Ramsey planner keeps tau at
// zero and the allocation is the social planner's. What the case exercises is the planner's
// condition for a PREDETERMINED stock: k(+1) is chosen at t, so its condition holds in
// expectation at t, and the private Euler is a constraint the planner must carry.
var c n k tau a;
varexo e;
parameters beta alpha delta rho chi phi;
beta  = 0.99;
alpha = 0.33;
delta = 0.025;
rho   = 0.95;
chi   = 8;
phi   = 1;
predetermined_variables k;

model;
1/c = beta*(1/c(+1))*(alpha*exp(a(+1))*k(+1)^(alpha-1)*n(+1)^(1-alpha) + 1 - delta);
chi*n^phi*c = (1-tau)*(1-alpha)*exp(a)*k^alpha*n^(-alpha);
c + k(+1) = exp(a)*k^alpha*n^(1-alpha) + (1-delta)*k;
a = rho*a(-1) + e;
end;

initval;
n = 0.330881054416; k = 9.379954789951; c = 0.763215941854; tau = 0; a = 0;
end;

shocks;
var e; stderr 0.01;
end;

planner_objective(log(c) - chi*n^(1+phi)/(1+phi));
ramsey_model(planner_discount=0.99, instruments=(tau));
steady;
stoch_simul(order=1, irf=40, nograph);
