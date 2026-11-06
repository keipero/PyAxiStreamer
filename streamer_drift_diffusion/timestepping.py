"""既存の移流・反応・電子速度式に対応する陽解法の時間刻み制御。"""
from dataclasses import dataclass
import math

import numpy as np
from .kinetics import DENSITY_NAMES
from .transport import ElectronTransport


@dataclass(frozen=True)
class TimestepLimits:
    """物理的な時間上限と、安全係数・設定上限を適用した候補刻みを保持する。

    Parameters
    ----------
    cfl : float
        移流・拡散を合算したCFL上限 [s]。
    dielectric : float
        誘電緩和による上限 [s]。
    chemistry : float
        反応の割合制約による上限 [s]。
    dt : float
        安全係数と設定上限を適用した候補刻み [s]。
    limiting : str
        支配した条件名。cfl、dielectric、chemistry、diffusion、dt_max。
    diffusion : float, optional
        純拡散だけの陽的上限 [s]。損失が0ならmath.inf。 Default: ``math.inf``.

    Notes
    -----
    凍結dataclassで、フィールドは書き換えない。通常はtimestep_limits()で生成する。
    cflはdt_cfl_numberを含むが、各物理上限にはdt_safety_factorをまだ掛けていない。

    Examples
    --------
    >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
    >>> config = SimulationConfig(nr=2, nz=3, voltage=0., nmax=0.,
    ...                           background_ionization=0., t_end=0.)
    >>> state = initialize_simulation(config)
    >>> limits = timestep_limits(state, config)
    >>> isinstance(limits, TimestepLimits)
    True
    >>> limits.dt
    1e-12
    """
    cfl: float
    dielectric: float
    chemistry: float
    dt: float
    limiting: str
    diffusion: float = math.inf


class TimestepError(RuntimeError):
    """安定な更新を受理できない、または計算の停止条件を満たした場合の例外。

    Parameters
    ----------
    *args : object
        RuntimeErrorへ渡すエラー情報。通常は説明文字列。

    Notes
    -----
    密度や物理モデルの不正状態を隠して成功として返さないために用いる。

    Examples
    --------
    >>> str(TimestepError('unstable dt'))
    'unstable dt'
    """


def validate_time_config(config):
    """両モードに共通する時間・停止上限と刻み制御の設定を検査する。

    Parameters
    ----------
    config : SimulationConfig
        物理定数、電圧、時間刻み・安定条件を保持する設定。

    Returns
    -------
    None
        正常時は何も変更せず戻る。

    Raises
    ------
    ValueError
        終了時刻、刻み範囲、安全係数、出力間隔、更新上限、背景供給または組成比が不正。

    Notes
    -----
    t_endは有限かつ0以上、output_intervalは有限かつ正。
    max_stepsは1以上の整数、max_step_retriesは0以上の整数。
    固定モードでも設定値の妥当性を検査する。

    Examples
    --------
    >>> from streamer_drift_diffusion import SimulationConfig
    >>> validate_time_config(SimulationConfig(t_end=0.))
    """
    if config.t_end is None or not math.isfinite(config.t_end) or config.t_end < 0:
        raise ValueError("requires a finite t_end >= 0 [s]")
    maximum = config.dt if config.dt_max is None else config.dt_max
    for name, value in (("dt_max", maximum), ("dt_min", config.dt_min),
                        ("dt_growth_factor", config.dt_growth_factor)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if config.dt_min > maximum:
        raise ValueError("dt_min must not exceed dt_max")
    if config.dt_growth_factor < 1:
        raise ValueError("dt_growth_factor must be >= 1")
    for name in ("dt_safety_factor", "dt_cfl_number", "dt_chemistry_fraction"):
        value = getattr(config, name)
        if not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"{name} must be in (0, 1]")
    for name, minimum in (("max_step_retries", 0), ("max_steps", 1)):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    interval = config.output_interval
    if interval is None or not math.isfinite(interval) or interval <= 0:
        raise ValueError("output_interval must be finite and positive")
    if not math.isfinite(config.background_ionization) or config.background_ionization < 0:
        raise ValueError("background_ionization must be finite and nonnegative")
    fractions = (config.nitrogen_fraction, config.oxygen_fraction)
    if any(not math.isfinite(x) or x < 0 for x in fractions) or not math.isclose(sum(fractions), 1., rel_tol=0., abs_tol=1e-12):
        raise ValueError("gas fractions must be nonnegative and sum to 1")


def validate_densities(state):
    """全粒子密度の有限性・非負性と電極内のゼロ条件を検査する。

    Parameters
    ----------
    state : SimulationState
        共有格子と計算配列。中心配列は(nr,nz)、面配列は方向別の全面形状。

    Returns
    -------
    None
        正常時は何も変更せず戻る。

    Raises
    ------
    TimestepError
        密度が負・非有限、または電極セルで非ゼロ。

    Notes
    -----
    負密度をクリップせず、異常を呼出側へ通知する。

    Examples
    --------
    >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
    >>> config = SimulationConfig(nr=2, nz=3, voltage=0., nmax=0.,
    ...                           background_ionization=0., t_end=0.)
    >>> state = initialize_simulation(config)
    >>> validate_densities(state)
    """
    for name in DENSITY_NAMES:
        density = getattr(state, name)
        if not np.all(np.isfinite(density)) or np.any(density < 0):
            raise TimestepError(f"invalid density: {name}")
        if np.any(density[~state.gas_mask] != 0.):
            raise TimestepError(f"density inside electrode: {name}")


def dielectric_limit(state, config):
    """面の移動度・電子密度から誘電緩和時間の上限を返す。

    Parameters
    ----------
    state : SimulationState
        共有格子と計算配列。中心配列は(nr,nz)、面配列は方向別の全面形状。
    config : SimulationConfig
        物理定数、電圧、時間刻み・安定条件を保持する設定。

    Returns
    -------
    float
        時間上限 [s]。有効な移動度×電子密度の最大が0ならmath.inf。

    Notes
    -----
    面の移動度と隣接する最大電子密度からepsilon0/(e*mu*ne)を評価する。
    移動度が全て0ならmath.inf。孤立した0面は有効面の最大移動度を使う。
    低電場の特異性は正則化せず、この評価規約だけでF→0の厳密な安定性を保証しない。

    Examples
    --------
    >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
    >>> config = SimulationConfig(nr=2, nz=3, voltage=0., nmax=0.,
    ...                           background_ionization=0., t_end=0.)
    >>> state = initialize_simulation(config)
    >>> dielectric_limit(state, config)
    inf
    """
    maximum_mobility = max(float(np.max(state.mobility_r)), float(np.max(state.mobility_z)))
    if maximum_mobility == 0.:
        return math.inf
    electrons = np.where(state.gas_mask, state.ne, 0.)
    density = np.maximum(*state.grid.adjacent(electrons))
    density[state.grid.area == 0.] = 0.
    mobility = state.grid.pack_faces(state.mobility_r, state.mobility_z)
    maximum = float(np.max(np.where(mobility > 0., mobility, maximum_mobility) * density))
    maximum *= config.elementary_charge / config.vacuum_permittivity
    return 1. / maximum if maximum > 0 else math.inf


def chemistry_limit(state, config):
    """反応速度から、密度変化の割合制約に対応する時間上限を返す。

    Parameters
    ----------
    state : SimulationState
        共有格子と計算配列。中心配列は(nr,nz)、面配列は方向別の全面形状。
    config : SimulationConfig
        物理定数、電圧、時間刻み・安定条件を保持する設定。

    Returns
    -------
    float
        時間上限 [s]。対象の反応速度が全て0ならmath.inf。

    Notes
    -----
    電子の生成・消費、中性粒子の消費を指定割合以下に抑える。
    生成と消費の相殺は使わず、気体セルだけを評価する。反応が全て0ならmath.inf。
    Canteraの反応機構を含む係数表では、密度更新と同じ反応次数と化学量論を使う。

    Examples
    --------
    >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
    >>> config = SimulationConfig(nr=2, nz=3, voltage=0., nmax=0.,
    ...                           background_ionization=0., t_end=0.)
    >>> state = initialize_simulation(config)
    >>> chemistry_limit(state, config)
    inf
    """
    if state.chemistry is not None and state.chemistry.mechanism is not None:
        maximum = state.chemistry.mechanism.maximum_frequency(state)
        return config.dt_chemistry_fraction / maximum if maximum > 0 else math.inf
    k_o2 = state.k1 if state.k1_o2 is None else state.k1_o2
    ionization = (state.k1*(state.n2 + state.o2) if state.k1_o2 is None
                  else state.k1*state.n2 + k_o2*state.o2)
    electron_rate = ionization + state.k2 * state.o2 + state.k3 * state.o2**2
    nitrogen_rate = state.k1 * state.ne
    oxygen_rate = (k_o2 + state.k2 + state.k3 * state.o2) * state.ne
    maximum = max(float(np.max(x[state.gas_mask])) for x in (electron_rate, nitrogen_rate, oxygen_rate))
    return config.dt_chemistry_fraction / maximum if maximum > 0 else math.inf


def timestep_limits(state, config, *, transport=None):
    """物理上限へ安全係数と設定上限を適用し、候補刻みを返す。

    Parameters
    ----------
    state : SimulationState
        共有格子と計算配列。中心配列は(nr,nz)、面配列は方向別の全面形状。
    config : SimulationConfig
        物理定数、電圧、時間刻み・安定条件を保持する設定。
    transport : ElectronTransport, optional
        再利用する輸送オブジェクト。Noneなら生成する。現在の速度・Dでprepare()を呼び、
        求めた面係数を次のadvance_density(..., prepared=True)でも共用できる。

    Returns
    -------
    TimestepLimits
        各上限、選択済みdt [s]、支配条件をまとめた値。

    Raises
    ------
    TimestepError
        いずれかの物理上限がNaNまたは0以下。
    ValueError
        輸送の格子・電極マスク・面配列形状またはDが不正。
    FloatingPointError
        輸送係数の準備で非有限の面速度や不正な演算を検出。

    Notes
    -----
    CFL、誘電緩和、反応、純拡散の最小へdt_safety_factorを掛ける。
    適応モードはdt_max（Noneならdt）、固定モードはdtで上限を設ける。
    イベントへの短縮はrun_simulation()が別途行う。

    Examples
    --------
    >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
    >>> config = SimulationConfig(nr=2, nz=3, voltage=0., nmax=0.,
    ...                           background_ionization=0., t_end=0.)
    >>> state = initialize_simulation(config)
    >>> transport = ElectronTransport(state, config.delta)
    >>> limits = timestep_limits(state, config, transport=transport)
    >>> limits.dt == config.dt
    True
    >>> limits.limiting
    'dt_max'
    >>> dt, retries = advance_density(state, config, limits.dt, transport=transport, prepared=True)
    >>> dt == config.dt, retries
    (True, 0)
    """
    transport = ElectronTransport(state, config.delta) if transport is None else transport
    transport.prepare(state)
    values: dict[str, float] = {"cfl": config.dt_cfl_number*transport.transport_limit,
                                "dielectric": dielectric_limit(state, config),
                                "chemistry": chemistry_limit(state, config),
                                "diffusion": transport.diffusion_limit}
    if any(math.isnan(x) or x <= 0 for x in values.values()):
        raise TimestepError("invalid timestep constraint")
    limiting = min(values, key=lambda name: values[name])
    dt = config.dt_safety_factor * values[limiting]
    maximum = config.dt if not config.adaptive_timestep or config.dt_max is None else config.dt_max
    if dt >= maximum:
        dt, limiting = maximum, "dt_max"
    return TimestepLimits(**values, dt=dt, limiting=limiting)


def advance_density(state, config, dt, *, transport=None, prepared=False):
    """共通の輸送・反応更新を試み、受理した密度と流出数を反映する。

    Parameters
    ----------
    state : SimulationState
        共有格子と計算配列。中心配列は(nr,nz)、面配列は方向別の全面形状。
    config : SimulationConfig
        物理定数、電圧、時間刻み・安定条件を保持する設定。
    dt : float
        開始時の試行幅 [s]。固定モードではこの幅を自動縮小しない。
    transport : ElectronTransport, optional
        再利用する輸送オブジェクト。Noneならこの呼出し用に生成する。
    prepared : bool, optional
        Trueなら指定transportが同じstateの現在の速度・Dで準備済みであることを前提とする。
        既定Falseでは試行前にprepare()する。再試行では同じ係数を使い、密度だけを復元する。

    Returns
    -------
    actual_dt : float
        受理した刻み [s]。
    retries : int
        棄却して再試行した回数。固定モードでは常に0。

    Raises
    ------
    TimestepError
        固定更新の失敗、再試行上限、または半減後の刻みがdt_min未満。
    ValueError
        輸送へ渡す刻み、格子・面配列・電極マスク、または拡散係数が不正。
        この例外では再試行しないため、検査済みの状態・設定を渡す。
    FloatingPointError
        試行前の係数準備で非有限の面速度や不正な演算を検出。この準備は再試行しない。

    Notes
    -----
    開始時の全密度を退避し、輸送後・反応後の非負性と物理制約を検査する。
    固定は失敗時に復元して停止、適応だけが半減・再試行する。
    流出数は受理時にだけ累積し、試行中に電場・係数は再計算しない。
    電子密度はstate.ne_varと輸送が同じ正本を参照する。各試行でFiPy旧値を更新し、
    棄却時はne_var.setValue()で復元する。他の密度も復元し、派生式のキャッシュも整合させる。
    この関数は時刻を進めない。更新後の電場準備は呼出側で行う。

    Examples
    --------
    >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
    >>> config = SimulationConfig(nr=2, nz=3, voltage=0., nmax=0.,
    ...                           background_ionization=0., t_end=0.)
    >>> state = initialize_simulation(config)
    >>> actual_dt, retries = advance_density(state, config, config.dt)
    >>> actual_dt == config.dt, retries
    (True, 0)
    """
    from .reactions import apply_reactions

    transport = ElectronTransport(state, config.delta) if transport is None else transport
    if not prepared:
        transport.prepare(state)
    saved = {name: getattr(state, name).copy() for name in DENSITY_NAMES}
    max_retries = config.max_step_retries if config.adaptive_timestep else 0
    for retries in range(max_retries + 1):
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                if np.any(state.de) and dt > transport.transport_limit * (1+1e-12):
                    raise TimestepError("explicit drift-diffusion constraint")
                losses = transport.step(state, dt, prepared=True)
                validate_densities(state)
                # 開始時に安全係数を適用済み。試行後は元の制約を越えた場合に拒否する。
                # 両端へ同じ安全係数を要求すると微小な密度増加だけで毎回半減してしまう。
                if dt > chemistry_limit(state, config) * (1 + 1e-12):
                    raise TimestepError("chemistry constraint after transport")
                apply_reactions(state, config, dt=dt)
                validate_densities(state)
                if dt > dielectric_limit(state, config) * (1 + 1e-12):
                    raise TimestepError("dielectric constraint after reactions")
            state.electron_losses[:] += losses
            return dt, retries
        except (TimestepError, FloatingPointError) as exc:
            state.ne_var.setValue(saved['ne'].ravel(order='F'))
            for name in DENSITY_NAMES[1:]:
                getattr(state, name)[:] = saved[name]
            if retries == max_retries or dt * 0.5 < config.dt_min:
                raise TimestepError(f"step rejected at dt={dt:.6e}: {exc}") from exc
            dt *= 0.5
