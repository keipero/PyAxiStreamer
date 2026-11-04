"""固定刻みの互換計算と、実時刻を管理する適応刻みの実行ループ。"""
from pathlib import Path
from contextlib import nullcontext
import math

import numpy as np

from .simulation_types import SimulationConfig
from .initial_conditions import initialize_simulation
from .electrostatics import calculate_charge_density, calculate_potential, calculate_electric_field
from .transport_coefficients import calculate_transport_coefficients
from .reactions import calculate_reaction_coefficients, apply_reactions
from .transport import ElectronTransport
from .voltage_waveforms import load_voltage_waveform, applied_voltage
from .output import write_c_vector, write_c_matrix, write_step_output, AdaptiveOutput
from .timestepping import TimestepError, timestep_limits, transport_limit, try_density_step, validate_densities


def prepare_step(state, config, time, *, waveform=None, verbose=True):
    """現在の密度・時刻から電荷源・電場・速度・反応係数を準備する。"""
    calculate_charge_density(state, config)
    voltage = calculate_potential(state, config, time=time, waveform=waveform, verbose=verbose)
    calculate_electric_field(state, config, voltage=voltage)
    calculate_transport_coefficients(state, config)
    calculate_reaction_coefficients(state)


def advance_step(state, config, time, *, waveform=None, verbose=True, transport=None):
    """
    電位・電場・電子輸送・反応を計算し、状態を1step更新する。

    Args:
        state (SimulationState): 更新対象の計算状態。電位、電場、速度、
            反応係数、粒子密度などの配列をその場で更新する。
        config (SimulationConfig): 時間刻み、印加電圧、Poisson反復条件、
            背景電離などの計算設定。
        time (float): このstepの印加電圧を評価する時刻 [s]。
        waveform (VoltageWaveform, optional): 読み込み済みのインパルス波形。
            Noneの場合、インパルスモードではconfig.voltage_waveform_file
            から読み込む。DCモードでは使用しない。既定値はNone。
        verbose (bool, optional): Trueの場合、Poisson反復の進捗を
            config.poisson_progress_everyに従って表示する。既定値はTrue。
        transport (ElectronTransport, optional): 再利用するFiPy輸送オブジェクト。
            省略時はこの呼出し用に生成する。

    Returns:
        None: 計算結果はstateの配列に反映する。

    Notes:
        この関数はconfig.dtで1回更新する固定刻みの入口。
        適応刻みの選択・受理・再試行はrun_simulation()で管理する。
        電荷源、電位、電場、電子ドリフト速度、反応係数の順に計算し、
        続いて電子移流と反応による密度更新をconfig.dtの時間幅で行う。
        背景電離は毎stepの密度加算量として適用し、dtを掛けない。
        電位・電場は密度更新前、粒子密度は更新後の値となる。
        時刻の加算、CSV保存、stepの進捗表示は呼び出し側で行う。
    """
    prepare_step(state, config, time, waveform=waveform, verbose=verbose)
    if np.any(state.de) and config.dt > transport_limit(state, config):
        raise TimestepError("fixed dt exceeds drift-diffusion CFL; reduce dt or use adaptive_timestep")
    transport = ElectronTransport(state, config.delta) if transport is None else transport
    state.electron_losses[:] += transport.step(state, config.dt)
    apply_reactions(state, config)


def run_simulation(config=None, *, output_dir="outputdata", write_output=True, verbose=True):
    """
    初期状態から時間発展を実行し、最終状態を返す。

    Args:
        config (SimulationConfig, optional): 格子、初期密度、印加電圧、
            時間刻み、終了stepなどの計算設定。Noneの場合は
            SimulationConfig()の既定値を使う。既定値はNone。
            adaptive_timestep=Trueの場合はt_end [s]の指定が必要。
            num_threadsは初期化時の単位電位求解の前にNumbaへ適用する。
        output_dir (str or os.PathLike, optional): CSVの保存先。
            相対パスは現在の作業ディレクトリを基準とする。
            出力有効時は必要な親フォルダも作成する。既定値は"outputdata"。
        write_output (bool, optional): Trueの場合、座標と指定間隔の
            物理量をCSVに保存する。Falseの場合は出力フォルダの作成と
            CSV保存を行わない。既定値はTrue。
        verbose (bool, optional): Trueの場合、config.write_everyごとに
            stepと時刻 [ns]、インパルスモードでは印加電圧 [V] も表示する。
            Poisson反復の進捗表示も有効にする。既定値はTrue。

    Returns:
        SimulationState: 最終stepの更新を終えた計算状態。
            固定モードの電位・電場は密度更新前、粒子密度は更新後の値。
            適応モードでは電場を再計算し、全てt_endの状態を返す。

    Raises:
        ValueError: 設定が不正、またはnum_threadsがNumbaの利用上限を超える場合。
        TimestepError: 適応モードで必要刻みがdt_min未満、再試行失敗、
            受理step上限、非有限値、Poisson未収束などが生じた場合。
        FileExistsError: 適応出力先に既存CSVがあり、混在を防ぐ場合。
    """
    config = SimulationConfig() if config is None else config
    state = initialize_simulation(config)
    transport = ElectronTransport(state, config.delta)
    waveform = load_voltage_waveform(config.voltage_waveform_file) if config.voltage_mode == "impulse" else None
    output_dir = Path(output_dir)
    if config.adaptive_timestep:
        return _run_adaptive(state, config, waveform, output_dir, write_output, verbose, transport)
    if write_output:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_c_vector(output_dir / "rh.csv", state.rh)
        write_c_vector(output_dir / "zh.csv", state.zh)
        write_c_matrix(output_dir / "gas_mask.csv", state.gas_mask)
    time = 0.0

    # ループのステップ数は0からconfig.max_stepまで
    for step in range(config.max_step + 1):
        # 1ステップ進める
        advance_step(state, config, time, waveform=waveform, verbose=verbose, transport=transport)
        if step % config.write_every == 0:
            if write_output:
                write_step_output(output_dir, step, state)
            if verbose:
                print(f"step = {step}, Time = {time * 1e9:.4f} ns")
                if config.voltage_mode == "impulse":
                    print(f"  Applied voltage = {applied_voltage(config, time, waveform=waveform):.6e} V")
        time += config.dt
    return state


def _prepare_adaptive(state, config, time, waveform, verbose):
    """適応計算の係数準備と有限性検査。受理状態以外からは呼び出さない。"""
    validate_densities(state)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        prepare_step(state, config, time, waveform=waveform, verbose=verbose)
    for name in ("rho", "phi", "ex", "ey", "abs_e", "vex", "vey", "k1", "k2", "k3"):
        if not np.all(np.isfinite(getattr(state, name))):
            raise TimestepError(f"nonfinite prepared field: {name}")


def _same_time(left, right):
    """有限な2時刻が8 ULP以内か判定する。単位は秒。

    ULPはその時刻付近の浮動小数点の間隔。固定の絶対誤差やdt_minを
    許容幅に使わず、初期時刻付近の小さい物理イベントも区別する。
    """
    if not (math.isfinite(left) and math.isfinite(right)):
        return False
    return abs(left - right) <= 8 * max(math.ulp(left), math.ulp(right))


def _run_adaptive(state, config, waveform, output_dir, write_output, verbose, transport):
    """実時刻で前進し、更新の受理後に電場も同じ時刻へ揃えて保存する。

    初期状態を出力番号0へ保存し、最終時刻も必ず保存する。
    max_step・write_everyの従来のstep解釈は固定モードだけに適用する。
    """
    interval = config.write_every * config.dt if config.output_interval is None else config.output_interval
    time, previous_dt, step, output_number = 0.0, None, 0, 1
    context = AdaptiveOutput(output_dir, config, state) if write_output else nullcontext(None)
    with context as output:
        try:
            _prepare_adaptive(state, config, time, waveform, verbose)
            if output:
                output.save_frame(step, time, 0.)
            while time < config.t_end:
                if step >= config.adaptive_max_steps:
                    raise TimestepError(f"adaptive_max_steps={config.adaptive_max_steps} reached at t={time:.9e}")
                limits = timestep_limits(state, config)
                if limits.dt < config.dt_min:
                    raise TimestepError(f"{limits.limiting} requires dt={limits.dt:.6e} < dt_min at t={time:.9e}")
                dt, limiting = limits.dt, limits.limiting
                if previous_dt is not None and dt > previous_dt * config.dt_growth_factor:
                    dt, limiting = previous_dt * config.dt_growth_factor, "growth"
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
                dt, retries = try_density_step(state, config, dt, transport=transport)
                advanced_time = time + dt
                # 使用dtは安定条件内のまま。時計だけをイベント時刻へ揃え、
                # 丸め誤差分について密度更新やstep加算を繰り返さない。
                time = event if _same_time(advanced_time, event) else advanced_time
                step += 1
                _prepare_adaptive(state, config, time, waveform, verbose)
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
            raise TimestepError(f"adaptive simulation stopped at t={time:.9e}, step={step}: {exc}") from exc
    return state
