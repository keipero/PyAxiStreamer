"""python -m streamer_drift_diffusion の入口。"""
import argparse

from .simulation_types import SimulationConfig, PROJECT_ROOT
from .simulation import run_simulation


def main(argv=None):
    """コマンドライン引数を設定へ変換し、シミュレーションを実行する。

    Parameters
    ----------
    argv : list of str, optional
        引数列。Noneならsys.argv[1:]を使用する。

    Returns
    -------
    int
        正常終了時は0。

    Raises
    ------
    SystemExit
        --help、未知のオプション、または実行時エラーをargparseが報告する場合。

    Notes
    -----
    --t-end、--output-interval、--max-stepsは両モード共通。
    --adaptive-timestepを指定すると安全刻みの選択と再試行を有効にする。

    Examples
    --------
    >>> main(['--nr', '2', '--nz', '3', '--t-end', '0', '--no-output', '--quiet'])
    0
    """
    defaults = SimulationConfig()
    parser = argparse.ArgumentParser(description="Axisymmetric streamer drift/reaction simulation")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputdata"))
    parser.add_argument("--voltage-mode", choices=("dc", "impulse"), default=defaults.voltage_mode)
    parser.add_argument("--voltage", type=float, default=defaults.voltage)
    parser.add_argument("--impulse-peak", type=float, default=defaults.impulse_peak_voltage)
    parser.add_argument("--impulse-time-scale", type=float, default=defaults.impulse_time_scale)
    parser.add_argument("--waveform", default=defaults.voltage_waveform_file)
    parser.add_argument("--nr", type=int, default=defaults.nr)
    parser.add_argument("--nz", type=int, default=defaults.nz)
    parser.add_argument("--poisson-tolerance", type=float, default=defaults.poisson_tolerance, help="Relative Poisson RHS residual tolerance")
    parser.add_argument("--delta", type=float, default=defaults.delta, help="Uniform cell spacing [m]")
    parser.add_argument("--electrode-gap", type=float, default=defaults.electrode_gap, help="Electrode tip height above ground [m]")
    parser.add_argument("--electron-diffusion", type=float, default=defaults.electron_diffusion,
                        help="Constant coefficient [m^2/s] for tables without BOLOS transport or empirical mode")
    parser.add_argument("--chemistry-table", help="JSON electron transport and reaction coefficients")
    parser.add_argument("--photoionization-efficiency", type=float, default=defaults.photoionization_efficiency,
                        help="Photon production factor before collisional quenching")
    parser.add_argument("--photoionization-workers", type=int, choices=(1, 2, 3),
                        default=defaults.photoionization_workers, help="Helmholtz solve threads")
    parser.add_argument("--no-output", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--adaptive-timestep", action="store_true", default=defaults.adaptive_timestep, help="Use adaptive explicit timesteps")
    parser.add_argument("--t-end", type=float, default=defaults.t_end, help="End time [s], common to fixed and adaptive modes")
    parser.add_argument("--dt", type=float, default=defaults.dt, help="Fixed dt / reference dt for background supply [s]")
    parser.add_argument("--dt-max", type=float, default=defaults.dt_max, help="Maximum adaptive dt [s], defaults to --dt")
    parser.add_argument("--dt-min", type=float, default=defaults.dt_min)
    parser.add_argument("--dt-safety-factor", type=float, default=defaults.dt_safety_factor)
    parser.add_argument("--dt-cfl-number", type=float, default=defaults.dt_cfl_number)
    parser.add_argument("--dt-chemistry-fraction", type=float, default=defaults.dt_chemistry_fraction)
    parser.add_argument("--dt-growth-factor", type=float, default=defaults.dt_growth_factor)
    parser.add_argument("--max-step-retries", type=int, default=defaults.max_step_retries)
    parser.add_argument("--max-steps", type=int, default=defaults.max_steps)
    parser.add_argument("--output-interval", type=float, default=defaults.output_interval, help="Output interval [s]")
    args = parser.parse_args(argv)
    config = SimulationConfig(
        voltage_mode=args.voltage_mode, voltage=args.voltage,
        impulse_peak_voltage=args.impulse_peak, impulse_time_scale=args.impulse_time_scale,
        voltage_waveform_file=args.waveform, nr=args.nr, nz=args.nz,
        electrode_gap=args.electrode_gap, delta=args.delta,
        electron_diffusion=args.electron_diffusion, chemistry_table=args.chemistry_table,
        photoionization_efficiency=args.photoionization_efficiency,
        photoionization_workers=args.photoionization_workers,
        poisson_tolerance=args.poisson_tolerance,
        adaptive_timestep=args.adaptive_timestep, t_end=args.t_end, dt=args.dt,
        dt_max=args.dt_max, dt_min=args.dt_min, dt_safety_factor=args.dt_safety_factor,
        dt_cfl_number=args.dt_cfl_number, dt_chemistry_fraction=args.dt_chemistry_fraction,
        dt_growth_factor=args.dt_growth_factor, max_step_retries=args.max_step_retries,
        max_steps=args.max_steps, output_interval=args.output_interval)
    try:
        run_simulation(config, output_dir=args.output, write_output=not args.no_output, verbose=not args.quiet)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
