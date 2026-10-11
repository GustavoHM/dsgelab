// A price LEVEL observed together with output: the level has a unit root, so the
// Kalman filter cannot start from a stationary distribution and must start
// DIFFUSE (Dynare's diffuse_filter; dsgelab's exact initialisation of Koopman and
// Durbin). Written for dsgelab's validation; MIT, like the package.
var y p pi;
varexo eps_pi eps_y;
parameters rho_pi rho_y;
rho_pi = 0.5;
rho_y  = 0.8;

model(linear);
pi = rho_pi*pi(-1) + eps_pi;
p = p(-1) + pi;
y = rho_y*y(-1) + 0.5*pi + eps_y;
end;

shocks;
var eps_pi; stderr 0.01;
var eps_y;  stderr 0.02;
end;

estimated_params;
rho_pi, beta_pdf, 0.5, 0.1;
rho_y, beta_pdf, 0.7, 0.1;
stderr eps_pi, inv_gamma_pdf, 0.01, inf;
stderr eps_y, inv_gamma_pdf, 0.02, inf;
end;

varobs p y;

estimation(datafile='unitroot_data.csv', diffuse_filter, mode_compute=0, mh_replic=0, smoother, nograph, plot_priors=0) y p pi;
