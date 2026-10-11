// A steady state that Newton cannot reach from the initval: the guess is good for
// delta = 0.02 and the model has delta = 0.9. homotopy_setup walks delta from one to
// the other. Written for dsgelab's validation; MIT.
var c k;
parameters alpha beta delta;
alpha = 0.33;
beta  = 0.99;
delta = 0.9;

model;
1/c = beta/c(+1)*(alpha*k^(alpha-1) + 1 - delta);
c + k = k(-1)^alpha + (1-delta)*k(-1);
end;

initval;
k = 30;
c = 2;
end;

homotopy_setup;
delta, 0.02, 0.9;
end;

steady(homotopy_mode=3, homotopy_steps=10);
