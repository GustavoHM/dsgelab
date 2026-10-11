// A steady_state_model that does not solve the model: the capital stock's closed form
// has 1/beta where it should have 1/beta - 1 + delta inside. Dynare refuses it, and so
// must dsgelab. Written for dsgelab's validation; MIT.
var c k a;
varexo e;
parameters alpha beta delta rho;
alpha = 0.33;
beta  = 0.99;
delta = 0.025;
rho   = 0.9;

model;
1/c = beta/c(+1)*(alpha*exp(a(+1))*k^(alpha-1) + 1 - delta);
c + k = exp(a)*k(-1)^alpha + (1-delta)*k(-1);
a = rho*a(-1) + e;
end;

steady_state_model;
a = 0;
k = (1/beta/alpha)^(1/(alpha-1));
c = k^alpha - delta*k;
end;

steady;
