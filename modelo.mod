// model — GERADO por dsgelab.to_dynare. Não editar à mão.
//
// Datação. Estoques predeterminados entram em `predetermined_variables`: a variável
// escrita sem defasagem é o estoque DISPONÍVEL em t, e X(+1) é o decidido em t.
// Processos exógenos seguem o dating PADRÃO do Dynare — a inovação realiza-se em t e
// move a variável em t —, e por isso a sua equação vai escrita um período atrás.
//   predeterminados : k
//   exógenos        : a

var k, a, c;
varexo eps_a;

predetermined_variables k;

parameters alpha, beta, rho, delta, gamma;
alpha = 0.36;
beta = 0.99;
rho = 0.95;
delta = 1.0;
gamma = 1.0;

model;
    c + k(+1) - a*k^alpha - k*(1 - delta) = 0;
    c^(-gamma) - beta*(1 - delta + a(+1)*alpha*k(+1)^(-1 + alpha))/c(+1)^gamma = 0;
    -eps_a - rho*log(a(-1)) + log(a) = 0;
end;

initval;
    k = 0.19948151091998417;
    a = 1.0;
    c = 0.3602309215154373;
    eps_a = 0;
end;

steady;
check;

shocks;
    var eps_a = 4.9000000000000005e-05;
end;

stoch_simul(order=2, irf=40, nograph);
