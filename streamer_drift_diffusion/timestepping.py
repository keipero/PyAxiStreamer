"""既存の移流・反応・電子速度式に対応する陽解法の時間刻み制御。"""
from dataclasses import dataclass
import math

import numpy as np

DENSITY_NAMES = ("ne", "n2", "o2", "n2p", "o2p", "om", "o2m")


@dataclass(frozen=True)
class TimestepLimits:
    """安全係数を掛ける前の各制約 [s] と、選択された安全な刻み。"""
    cfl: float
    dielectric: float
    chemistry: float
    dt: float
    limiting: str
    diffusion: float = math.inf


class TimestepError(RuntimeError):
    """安定な更新を受理できない、または停止条件を満たした場合の例外。"""


def validate_adaptive_config(config):
    """適応刻み専用の設定を検査する。固定モードの設定解釈は変えない。"""
    if config.t_end is None or not math.isfinite(config.t_end) or config.t_end < 0:
        raise ValueError("adaptive mode requires a finite t_end >= 0 [s]")
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
    for name, minimum in (("max_step_retries", 0), ("adaptive_max_steps", 1)):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    interval = config.write_every * config.dt if config.output_interval is None else config.output_interval
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("output_interval must be finite and positive")
    if not math.isfinite(config.background_ionization) or config.background_ionization < 0:
        raise ValueError("background_ionization must be finite and nonnegative")
    fractions = (config.nitrogen_fraction, config.oxygen_fraction)
    if any(not math.isfinite(x) or x < 0 for x in fractions) or not math.isclose(sum(fractions), 1., rel_tol=0., abs_tol=1e-12):
        raise ValueError("gas fractions must be nonnegative and sum to 1")


def validate_densities(state):
    """全密度の有限性・非負性を検査する。クリップせず、異常時は失敗する。"""
    for name in DENSITY_NAMES:
        density = getattr(state, name)
        if not np.all(np.isfinite(density)) or np.any(density < 0):
            raise TimestepError(f"invalid density: {name}")
        if np.any(density[~state.gas_mask] != 0.):
            raise TimestepError(f"density inside electrode: {name}")


def diffusion_loss(state, config):
    """拡散更新のセルごとの損失対角係数 [m^3/s]。拡散なしでは0を返す。"""
    coefficient = getattr(state, 'de', None)
    if coefficient is None or not np.any(coefficient):
        return 0.
    return state.grid.sum_faces(state.grid.diffusion_rates(coefficient, state.gas_mask))


def diffusion_limit(state, config):
    """純拡散の陽的上限 [s]。境界面の半セル距離も考慮する。"""
    gas = getattr(state, 'gas_mask', np.ones_like(state.volume, dtype=bool))
    maximum = float(np.max((diffusion_loss(state, config) / state.volume)[gas]))
    return 1./maximum if maximum > 0 else math.inf


def transport_limit(state, config):
    """ドリフトと拡散の損失対角を合算したCFL上限 [s]を返す。

    別々の上限の最小値だけでは同時更新の非負性を保証できないため、
    実際の全セル面の流出係数を合算する。拡散0では従来の移流CFLになる。
    """
    grid = state.grid
    rate = grid.pack_faces(state.vex, state.vey)*grid.area
    loss = grid.sum_faces(np.maximum(rate, 0.), np.maximum(-rate, 0.))
    loss += diffusion_loss(state, config)
    gas = getattr(state, 'gas_mask', np.ones_like(state.volume, dtype=bool))
    maximum = float(np.max((loss / state.volume)[gas]))
    return config.dt_cfl_number / maximum if maximum > 0 else math.inf


def dielectric_limit(state, config):
    """輸送と同じ各面の移動度・隣接する最大電子密度から誘電緩和を制限する。

    現行速度式に対応するmu = 3.74e24 * F**(-1/4) / Nを用いる。
    F=0で式が特異となるため、孤立した0面は有効面の最大移動度で評価する。
    全面の移動度が0ならドリフトは0で、この制約は無限大とする。
    この0点の扱いは有限な評価規約であり、F→0での厳密な安定性保証ではない。
    速度・反応式の正則化は行わない。
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
    """電子の生成・消費と中性粒子の消費を、指定した割合以下に抑える。

    生成と消費の相殺を使わず、正味源が小さい場合も速い反応を制限する。
    移流後にも再評価し、逐次更新で変わった電子密度へ対応する。
    """
    electron_rate = state.k1 * (state.n2 + state.o2) + state.k2 * state.o2 + state.k3 * state.o2**2
    nitrogen_rate = state.k1 * state.ne
    oxygen_rate = (state.k1 + state.k2 + state.k3 * state.o2) * state.ne
    maximum = max(float(np.max(x[state.gas_mask])) for x in (electron_rate, nitrogen_rate, oxygen_rate))
    return config.dt_chemistry_fraction / maximum if maximum > 0 else math.inf


def timestep_limits(state, config):
    """合算輸送CFL・誘電緩和・反応・純拡散の上限から候補を返す。"""
    values: dict[str, float] = {"cfl": transport_limit(state, config),
                                "dielectric": dielectric_limit(state, config),
                                "chemistry": chemistry_limit(state, config),
                                "diffusion": diffusion_limit(state, config)}
    if any(math.isnan(x) or x <= 0 for x in values.values()):
        raise TimestepError("invalid timestep constraint")
    limiting = min(values, key=lambda name: values[name])
    dt = config.dt_safety_factor * values[limiting]
    maximum = config.dt if config.dt_max is None else config.dt_max
    if dt >= maximum:
        dt, limiting = maximum, "dt_max"
    return TimestepLimits(**values, dt=dt, limiting=limiting)


def try_density_step(state, config, dt, *, transport=None):
    """準備済みの電場・係数を使って更新し、失敗した試行を完全に復元する。

    Returns:
        tuple[float, int]: 受理したdt [s]と拒否回数。

    Notes:
        開始状態の刻み選択では安全係数を掛ける。試行後には安全係数を
        重ねず、反応の割合上限と誘電緩和時間そのものを超えたら拒否する。
        復元するのは密度だけでよい。試行中に電場・係数は再計算しない。
    """
    from .transport import ElectronTransport
    from .reactions import apply_reactions

    transport = ElectronTransport(state, config.delta) if transport is None else transport
    saved = {name: getattr(state, name).copy() for name in DENSITY_NAMES}
    for retries in range(config.max_step_retries + 1):
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                if np.any(state.de) and dt > transport_limit(state, config)/config.dt_cfl_number * (1+1e-12):
                    raise TimestepError("explicit drift-diffusion constraint")
                losses = transport.step(state, dt)
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
            for name, values in saved.items():
                getattr(state, name)[:] = values
            if retries == config.max_step_retries or dt * 0.5 < config.dt_min:
                raise TimestepError(f"step rejected at dt={dt:.6e}: {exc}") from exc
            dt *= 0.5
