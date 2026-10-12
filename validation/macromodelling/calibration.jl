# A calibration equation in MacroModelling.jl: beta is whatever puts the capital-output
# ratio at 2.5 in the steady state. Writes the calibrated beta and the steady state to calibration.json,
# for validation/compare_calibration.py. Written for dsgelab's validation; MIT.
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

ss = get_steady_state(RBC, derivatives = false)
names = string.(MacroModelling.AxisKeys.axiskeys(ss, 1))
vals = vec(collect(ss))
cal = get_calibrated_parameters(RBC, values = true)
out = Dict("steady_state" => Dict(zip(names, vals)),
           "calibrated" => Dict(string(k) => v for (k, v) in cal))
println(out)
open(joinpath(@__DIR__, "calibration.json"), "w") do io
    JSON3.write(io, out)
end
