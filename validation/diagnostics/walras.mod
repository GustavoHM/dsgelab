// A broken model, on purpose: the Euler equation is missing and a resource constraint
// implied by two others (Walras' law) stands in its place, so the static Jacobian is
// singular with one collinear relationship; and q appears only one period ahead, so it
// is not present at the current period. Written for dsgelab's validation; MIT.
var y c i k a q;
varexo e;
parameters alpha delta rho;
alpha = 0.33;
delta = 0.025;
rho   = 0.9;

model;
y = exp(a)*k(-1)^alpha;
y = c + i;
k = (1-delta)*k(-1) + i;
c + i = exp(a)*k(-1)^alpha;
a = rho*a(-1) + e;
q(+1) = a;
end;

steady_state_model;
a = 0;
k = 10;
y = k^alpha;
i = delta*k;
c = y - i;
q = 0;
end;

model_diagnostics;
