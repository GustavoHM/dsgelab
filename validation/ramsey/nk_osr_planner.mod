// An optimal simple rule chosen on welfare: the New Keynesian model with a Taylor rule
// whose two coefficients maximise the planner's objective, unconditionally, at second
// order. Written for dsgelab's validation; MIT, like the package.
var x pi i u;
varexo e;
parameters beta sigma kappa lam rho phi_pi phi_u;
beta   = 0.99;
sigma  = 1;
kappa  = 0.1;
lam    = 0.25;
rho    = 0.5;
phi_pi = 1.5;
phi_u  = 0;

model(linear);
x = x(+1) - (1/sigma)*(i - pi(+1));
pi = beta*pi(+1) + kappa*x + u;
i = phi_pi*pi + phi_u*u;
u = rho*u(-1) + e;
end;

shocks;
var e; stderr 0.01;
end;

planner_objective -(pi^2 + lam*x^2)/2;
osr_params phi_pi phi_u;
osr(planner_discount=0.99, order=2, silent_optimizer);
