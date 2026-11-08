"""固定／適応刻みの共通ループ。同じ実時刻の密度・電場を保存する。"""
from contextlib import nullcontext
import math

import numpy as np

from .simulation_types import SimulationConfig
from .initial_conditions import initialize_simulation
from .electrostatics import calculate_charge_density, calculate_potential, calculate_electric_field
from .transport_coefficients import calculate_transport_coefficients
from .reactions import calculate_reaction_coefficients
from .transport import ElectronTransport
from .voltage_waveforms import load_voltage_waveform
from .output import SimulationOutput
from .timestepping import TimestepError, timestep_limits, advance_density, validate_densities


def prepare_step(state, config, time, *, waveform=None, verbose=True):
    """受理済みの密度から同一時刻の電場・係数を準備する。

    Parameters
    ----------
    state : SimulationState
        共有格子とセル中心・全面の計算配列。配列添字は[半径, 軸方向]。
    config : SimulationConfig
        格子寸法・物理定数・境界・刻み制御を保持する設定。変更しない。
    time : float
        評価時刻 [s]。密度が表す時刻と一致させる。
    waveform : VoltageWaveform, optional
        読み込み済み波形。Noneならインパルス時に設定ファイルを読む。
    verbose : bool, optional
        進捗表示の指定。現在のPoisson LU求解は反復進捗を表示しない。

    Returns
    -------
    None
        rho、phi、電場・速度・反応係数と有効時の光電離源をその場で更新する。

    Raises
    ------
    TimestepError
        密度が非有限・負、電極内部に粒子が存在する、または準備した場が非有限。
    FloatingPointError
        電場・物理係数の評価中にオーバーフローや不正演算が発生。
    RuntimeError
        Poisson求解が有限性・残差検査に失敗。
    ValueError
        評価時刻・電圧モード・波形データが不正。
    OSError
        波形ファイルの読込みが失敗。

    Notes
    -----
    粒子密度や累積流出数は更新しない。計算開始時と各受理更新後に呼び、
    次の輸送にも同じ電場・係数を再利用する。

    Examples
    --------
    >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
    >>> config = SimulationConfig(nr=2, nz=3, voltage=0., nmax=0.,
    ...                           background_ionization=0., t_end=0.)
    >>> state = initialize_simulation(config)
    >>> from streamer_drift_diffusion.simulation import prepare_step
    >>> prepare_step(state, config, 0., verbose=False)
    >>> float(state.abs_e.max())
    0.0
    """
    validate_densities(state)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        calculate_charge_density(state, config)
        voltage = calculate_potential(state, config, time=time, waveform=waveform, verbose=verbose)
        calculate_electric_field(state, config, voltage=voltage)
        calculate_transport_coefficients(state, config)
        calculate_reaction_coefficients(state)
        if state.reaction_rates is not None:
            impact = state.chemistry.mechanism.impact_ionization(state)
        else:
            k_o2 = state.k1 if state.k1_o2 is None else state.k1_o2
            impact = state.ne * (state.k1 * state.n2 + k_o2 * state.o2)
        state.photoionization.solve(impact)
    for name in ("rho", "phi", "ex", "ey", "abs_e", "vex", "vey", "k1", "k1_o2", "k2", "k3"):
        values = getattr(state, name)
        if values is not None and not np.all(np.isfinite(values)):
            raise TimestepError(f"nonfinite prepared field: {name}")


def _same_time(left, right):
    """有限な2時刻が、その時刻付近の8 ULP以内か判定する。

    Parameters
    ----------
    left, right : float
        比較する時刻 [s]。

    Returns
    -------
    bool
        両方が有限で差が8 ULP以内ならTrue。

    Notes
    -----
    固定の絶対許容値やdt_minを使わず、初期時刻付近の小さい物理イベントを区別する。

    Examples
    --------
    >>> _same_time(0., 0.)
    True
    >>> _same_time(float('inf'), float('inf'))
    False
    """
    if not (math.isfinite(left) and math.isfinite(right)):
        return False
    return abs(left - right) <= 8 * max(math.ulp(left), math.ulp(right))


def run_simulation(config=None, *, output_dir="outputdata", write_output=True, verbose=True):
    """固定／適応刻みの共通ループで、初期時刻から終了時刻まで計算する。

    Parameters
    ----------
    config : SimulationConfig, optional
        計算条件。Noneなら既定設定。t_end [s]とoutput_interval [s]は両モード共通。
    output_dir : str or os.PathLike, optional
        出力先。出力有効時は既存CSVのないフォルダを指定する。
    write_output : bool, optional
        Trueなら初期・指定間隔・最終状態と全受理更新の履歴を保存する。
    verbose : bool, optional
        Trueなら保存時刻に受理step番号と時刻を表示する。

    Returns
    -------
    SimulationState
        t_endに揃った密度・電位・電場。t_end=0なら密度更新前の初期状態。

    Raises
    ------
    ValueError
        格子、時間、係数または波形の設定が不正。
    FileExistsError
        出力先に既存CSVがあり、計算条件の混在を防ぐ場合。
    TimestepError
        安定刻みを受理できない、固定幅が不安定、更新数上限、または電場準備が失敗。
    RuntimeError
        Poisson求解の有限性・残差検査が失敗。
    FloatingPointError
        電場・物理係数の準備中にオーバーフローや不正演算が発生。
    OSError
        波形入力・計算出力の読み書きに失敗。

    Notes
    -----
    出力番号0はt=0の初期状態。全フレームで電位・電場・密度の時刻を揃える。
    固定刻みも出力・波形の折れ点・終端でだけ短縮し、不安定なら停止する。
    適応刻みだけが失敗した更新を復元・半減・再試行する。
    無出力でも同じイベントで刻みを分割するため、保存の有無で積分経路は変わらない。
    背景加算量はbackground_ionization * actual_dt / config.dt。
    光電離源は受理済み密度で評価し、次の更新と再試行に共用する。
    成功・失敗とも終了時にPoissonと光電離のLU分解を解放する。
    返した状態では電位、Poisson行列、直近の光電離源を保持する。

    Examples
    --------
    >>> from streamer_drift_diffusion import SimulationConfig
    >>> config = SimulationConfig(nr=2, nz=3, voltage=0., nmax=0.,
    ...                           background_ionization=0., t_end=1e-12)
    >>> state = run_simulation(config, write_output=False, verbose=False)
    >>> state.ne.shape
    (2, 3)
    >>> float(state.ne.max())
    0.0
    """
    config = SimulationConfig() if config is None else config
    state = initialize_simulation(config)
    transport = ElectronTransport(state, config.delta)
    waveform = load_voltage_waveform(config.voltage_waveform_file) if config.voltage_mode == "impulse" else None
    interval = config.output_interval
    time, previous_dt, step, output_number = 0.0, None, 0, 1
    context = SimulationOutput(output_dir, config, state) if write_output else nullcontext(None)
    with context as output:
        try:
            prepare_step(state, config, time, waveform=waveform, verbose=verbose)
            if output:
                output.save_frame(step, time, 0.)
            while time < config.t_end:
                if step >= config.max_steps:
                    raise TimestepError(f"max_steps={config.max_steps} reached at t={time:.9e}")
                limits = timestep_limits(state, config, transport=transport)
                if config.adaptive_timestep:
                    if limits.dt < config.dt_min:
                        raise TimestepError(f"{limits.limiting} requires dt={limits.dt:.6e} < dt_min at t={time:.9e}")
                    dt, limiting = limits.dt, limits.limiting
                    if previous_dt is not None and dt > previous_dt * config.dt_growth_factor:
                        dt, limiting = previous_dt * config.dt_growth_factor, "growth"
                else:
                    dt, limiting = config.dt, "fixed"
                unclipped_dt = dt
                next_output = output_number * interval
                event, event_kind = config.t_end, "t_end"
                # ほぼ同じ時刻は、終了→出力→波形の優先順位でまとめる。
                if next_output < event and not _same_time(next_output, event):
                    event, event_kind = next_output, "output"
                # 入力波形の折れ点を飛び越さず、短い外部パルスを見落とさない。
                if waveform is not None:
                    scaled_time = time / config.impulse_time_scale
                    index = np.searchsorted(waveform.times, scaled_time, side="right")
                    while index < len(waveform.times):
                        knot = float(waveform.times[index] * config.impulse_time_scale)
                        if knot > time and not _same_time(knot, time):
                            if knot < event and not _same_time(knot, event):
                                event, event_kind = knot, "waveform"
                            break
                        index += 1
                remaining = event - time
                if dt >= remaining:
                    dt, limiting = remaining, event_kind
                if not np.isfinite(dt) or dt <= 0 or time + dt == time:
                    raise TimestepError(f"time cannot advance at t={time:.9e}, dt={dt}")
                # 出力・終端へ揃える最後の端数は、dt_minより小さくても安全側。
                if not config.adaptive_timestep and dt > limits.dt*(1+1e-12):
                    raise TimestepError(f"fixed dt exceeds {limits.limiting} stability constraint; reduce dt or use adaptive_timestep")
                dt, retries = advance_density(state, config, dt, transport=transport, prepared=True)
                advanced_time = time + dt
                # 使用dtは安定条件内のまま。時計だけをイベント時刻へ揃え、
                # 丸め誤差分について密度更新やstep加算を繰り返さない。
                time = event if _same_time(advanced_time, event) else advanced_time
                step += 1
                prepare_step(state, config, time, waveform=waveform, verbose=verbose)
                if output:
                    output.record_step(step, time, dt, limits,
                                       "retry" if retries else limiting, retries)
                at_output = _same_time(time, next_output)
                save = at_output or _same_time(time, config.t_end)
                if save:
                    if output:
                        output.save_frame(step, time, dt)
                    if verbose:
                        print(f"step = {step}, Time = {time * 1e9:.6f} ns, dt = {dt:.6e} s")
                    if at_output:
                        output_number += 1
                # 端数合わせだけで次の刻みを過度に縮めない。再試行時は縮小値を保持。
                previous_dt = dt if retries else unclipped_dt
        except (TimestepError, FloatingPointError, RuntimeError) as exc:
            if output:
                output.save_failure(step, time, exc)
            raise TimestepError(f"simulation stopped at t={time:.9e}, step={step}: {exc}") from exc
        finally:
            state.poisson.close()
            state.photoionization.close()
    return state
