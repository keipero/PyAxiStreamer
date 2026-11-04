# 移植元を直接実行し、全状態配列を17桁で書き出す比較専用スクリプト。
include(joinpath(@__DIR__, "..", "..", "streamer_julia", "src", "StreamerDriftDiffusion.jl"))
using .StreamerDriftDiffusion
using Printf

root = ARGS[1]
cases = (
    ("small_dc", SimulationConfig(nr=12, nz=10, max_step=20, z0=70e-6, cathode_radius=30e-6)),
    ("small_impulse", SimulationConfig(nr=12, nz=10, max_step=20, z0=70e-6,
        cathode_radius=30e-6, voltage_mode=:impulse, impulse_time_scale=2.0)),
    ("small_negative", SimulationConfig(nr=12, nz=10, max_step=20, z0=70e-6,
        cathode_radius=30e-6, voltage=-4000.0, poisson_omega=1.0)),
    ("standard_100", SimulationConfig(max_step=100)),
)
for (name, config) in cases
    dir = joinpath(root, name)
    state = run_simulation(config; output_dir=dir, verbose=false)
    for field in fieldnames(SimulationState)
        values = getproperty(state, field)
        open(joinpath(dir, "state_$(field).txt"), "w") do io
            if values isa Matrix
                for i in axes(values, 1), j in axes(values, 2)
                    @printf(io, "%.17e\n", values[i, j])
                end
            else
                for value in values
                    @printf(io, "%.17e\n", value)
                end
            end
        end
    end
    println("Julia reference complete: ", name)
end
