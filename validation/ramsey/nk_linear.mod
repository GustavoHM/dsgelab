// The textbook New Keynesian model (Gali 2015, ch. 5) under Ramsey commitment.
// Written for dsgelab's validation; MIT, like the package.
//
// The analytic answer is known: the targeting rule pi_t = -(lam/kappa)(x_t - x_{t-1})
// and the closed form for x_t. dsgelab is checked against both in tests/test_optimal.py;
// here Dynare's ramsey_model is the third witness.
var x pi i u;
varexo e;
parameters beta sigma kappa lam rho;
beta  = 0.99;
sigma = 1;
kappa = 0.1;
lam   = 0.25;
rho   = 0.5;

model;
x = x(+1) - (1/sigma)*(i - pi(+1));
pi = beta*pi(+1) + kappa*x + u;
u = rho*u(-1) + e;
end;

initval;
x = 0; pi = 0; i = 0; u = 0;
end;

shocks;
var e; stderr 1;
end;

planner_objective(-(pi^2 + lam*x^2)/2);
ramsey_model(planner_discount=0.99, instruments=(i));
steady;
stoch_simul(order=1, irf=40, nograph);
