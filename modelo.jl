# model — GERADO por dsgelab.to_macromodelling. Não editar à mão.
# Datação: um estoque predeterminado disponível em t é `x[-1]`; um salto em t é
# `x[0]`. As inovações são normais PADRÃO aqui, logo o desvio-padrão entra NA EQUAÇÃO.
#   predeterminados : k
#   exógenos        : a

using MacroModelling

@model model begin
    c[0] + k[0] - a[0]*k[-1]^alpha - k[-1]*(1 - delta) = 0
    c[0]^(-gamma) - beta*(1 - delta + a[1]*alpha*k[0]^(-1 + alpha))/c[1]^gamma = 0
    -sig_eps_a * eps_a[x] - rho*log(a[-1]) + log(a[0]) = 0
end

@parameters model begin
    alpha = 0.36
    beta = 0.99
    rho = 0.95
    delta = 1.0
    gamma = 1.0
    sig_eps_a = 0.007
end

