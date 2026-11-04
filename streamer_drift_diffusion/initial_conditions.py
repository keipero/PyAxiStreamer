"""設定検査、独立した配列確保、電荷中性のGaussian初期分布。"""
from dataclasses import fields
import math
from numbers import Integral, Real

import numpy as np

from .simulation_types import SimulationConfig, SimulationState, molecular_density
from .grid import initialize_grid
from .electrostatics import initialize_potential
from ._acceleration import set_num_threads
from .voltage_waveforms import validate_impulse_parameters
from .transport_coefficients import calculate_transport_coefficients


def validate_config(config):
    """初期化と時間発展に使う主要な設定値の範囲を検査する。

    Args:
        config (SimulationConfig): 格子数、時間刻み、初期分布の幅、
            電圧モード、Poisson反復条件などを保持する計算設定。

    Raises:
        ValueError: nr・nzが2以上、max_stepが0以上、write_everyが1以上の
            整数でない場合。これらの値にboolを指定した場合も送出する。
            num_threadsは1以上の整数とし、boolは指定できない。
            poisson_omegaが0より大きく2未満でない、voltage_modeが
            dc・impulse以外、またはvoltageが有限でない場合にも送出する。
            delta、temperature、sr0、sz0、cathode_radius、dt、
            gas_constant、vacuum_permittivity、poisson_toleranceは、
            いずれも有限かつ正である必要がある。
            impulseモードでは、振幅と時間倍率が
            validate_impulse_parameters()の条件を満たさない場合も送出する。

    Returns:
        None: 検査に成功した場合、そのまま呼び出し元へ戻る。

    Notes:
        設定値は変更しない。全フィールドを網羅する検査ではなく、
        固定モードではガス組成比の和や初期密度の範囲、波形ファイルの内容を
        検査しない。適応モードではtimestepping.validate_adaptive_config()で
        刻み設定・背景供給・ガス組成比も追加検査する。
    """
    for name, minimum in (("nr", 2), ("nz", 2), ("max_step", 0), ("write_every", 1),
                          ("num_threads", 1), ("poisson_max_iterations", 1)):
        value = getattr(config, name)
        if not isinstance(value, Integral) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not 0 < config.poisson_omega < 2:
        raise ValueError("poisson_omega must satisfy 0 < omega < 2")
    if config.voltage_mode not in ("dc", "impulse"):
        raise ValueError("voltage_mode must be dc or impulse")
    if not math.isfinite(config.voltage):
        raise ValueError("voltage must be finite")
    diffusion = config.electron_diffusion
    if (not isinstance(diffusion, Real) or isinstance(diffusion, bool)
            or not math.isfinite(diffusion) or diffusion < 0):
        raise ValueError("electron_diffusion must be finite and nonnegative [m^2/s]")
    cfl = config.dt_cfl_number
    if not isinstance(cfl, Real) or isinstance(cfl, bool) or not 0 < cfl <= 1:
        raise ValueError("dt_cfl_number must be in (0, 1]")
    # Python版で追加: 幅ゼロ・無効な反復条件による計算不能を初期化時に検出。
    for name in ("delta", "temperature", "sr0", "sz0", "cathode_radius", "dt",
                 "gas_constant", "vacuum_permittivity", "poisson_tolerance", "electrode_gap"):
        value = getattr(config, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if config.voltage_mode == "impulse":
        validate_impulse_parameters(config)
    if config.adaptive_timestep:
        from .timestepping import validate_adaptive_config
        validate_adaptive_config(config)


def initialize_simulation(config=None):
    """格子・電極マスク・Poisson係数・初期粒子分布を生成する。

    Args:
        config (SimulationConfig, optional): Noneなら既定値。計算条件はここに集約。

    Returns:
        SimulationState: 独立した状態配列。座標等は1次元、
            密度・電位・中心電場等は(nr,nz)。半径面の電場・速度・移動度は
            (nr+1,nz)、軸方向面は(nr,nz+1)。electron_lossesは長さ3。

    Raises:
        ValueError: 不正設定、利用できないスレッド数、気体セルのない格子。
        RuntimeError: 単位電位のPoisson反復が収束しない場合。

    Notes:
        スレッド数を適用し、1 V印加時の離散ラプラス解を初期化時だけ求める。
        初期phi/pphi/lphiと電場は0。時間発展時にunit_potentialを電圧倍する。
        初期の中性粒子・電荷中性Gaussian・背景供給は気体セルに限定する。
        電極内の全密度と累積流出電子数は0。gas_maskはbool、他はfloat64。"""
    config = SimulationConfig() if config is None else config
    validate_config(config)
    set_num_threads(config.num_threads)
    arrays = initialize_grid(config)
    if not np.any(arrays['gas_mask']):
        raise ValueError("grid contains no gas cells; check electrode_gap, cathode_radius and delta")
    for name in ("er_face", "vex", "mobility_r"):
        arrays[name] = np.zeros((config.nr + 1, config.nz), dtype=np.float64)
    for name in ("ez_face", "vey", "mobility_z"):
        arrays[name] = np.zeros((config.nr, config.nz + 1), dtype=np.float64)
    arrays["radial_boundary"] = np.zeros(config.nz, dtype=np.float64)
    arrays["upper_boundary"] = np.zeros(config.nr, dtype=np.float64)
    for field in fields(SimulationState):
        if field.name not in arrays:
            arrays[field.name] = np.zeros((config.nr, config.nz), dtype=np.float64)
    state = SimulationState(**arrays)
    initialize_potential(state, config)
    r, z = state.rh[:, None], state.zh[None, :]
    # 初期電子・イオン　初期分布をGaussianで生成し、局所的な電荷中性を保つ
    gaussian = np.exp(-(r * r) / (2 * config.sr0 * config.sr0)
                      - ((z - config.z0) * (z - config.z0)) / (2 * config.sz0 * config.sz0))
    mol = molecular_density(config)
    state.n2[:] = config.nitrogen_fraction * mol * state.gas_mask
    state.o2[:] = config.oxygen_fraction * mol * state.gas_mask
    state.n2p[:] = config.nitrogen_fraction * config.nmax * gaussian * state.gas_mask
    state.o2p[:] = config.oxygen_fraction * config.nmax * gaussian * state.gas_mask
    state.ne[:] = state.n2p + state.o2p
    calculate_transport_coefficients(state, config)
    return state
