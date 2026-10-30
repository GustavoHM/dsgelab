using MacroModelling, JSON3

@model RBC begin
    1 / c[0] = β / c[1] * (α * exp(z[1]) * k[0]^(α - 1) + 1 - δ)
    c[0] + k[0] = exp(z[0]) * k[-1]^α + (1 - δ) * k[-1]
    z[0] = ρ * z[-1] + σ * ϵ[x]
    q[0] = exp(z[0]) * k[-1]^α
end

@parameters RBC begin
    α = 0.5
    δ = 0.02
    ρ = 0.2
    σ = 0.01
    k[ss] / q[ss] = 2.5 | β
end

simulation = simulate(RBC, algorithm = :pruned_second_order, periods = 30)
data = simulation([:k], :, :simulate)
ll = get_loglikelihood(RBC, data, RBC.parameter_values;
                       filter = :inversion, algorithm = :pruned_second_order)
shocks = get_estimated_shocks(RBC, data;
                              algorithm = :pruned_second_order, filter = :inversion)
out = Dict("loglik" => ll,
           "shocks" => Array(shocks),
           "data" => Array(data),
           "parameters" => RBC.parameter_values)
open(joinpath(@__DIR__, "inversion.json"), "w") do io
    JSON3.write(io, out)
end
println(out)
