"""設定検査、独立した配列確保、電荷中性のGaussian初期分布。"""
from dataclasses import fields
import math
from numbers import Integral, Real

import numpy as np
from fipy import CellVariable

from .simulation_types import SimulationConfig, SimulationState, molecular_density
from .grid import initialize_grid
from .electrostatics import PoissonSolver
from .voltage_waveforms import validate_impulse_parameters
from .transport_coefficients import calculate_transport_coefficients


def validate_config(config):
    """初期化と時間発展に使う主要な設定値の範囲を検査する。

    Parameters
    ----------
    config : SimulationConfig
        格子数、時間刻み、初期分布の幅、
        電圧モード、Poisson求解条件などを保持する計算設定。

    Returns
    -------
    None
        検査に成功した場合、そのまま呼び出し元へ戻る。

    Raises
    ------
    ValueError
        nr・nzが2以上の整数でない場合。boolも指定できない。
        voltage_modeがdc・impulse以外、またはvoltageが有限でない場合。
        delta、temperature、sr0、sz0、cathode_radius、dt、
        gas_constant、vacuum_permittivity、poisson_tolerance、electrode_gapは、
        いずれも有限かつ正である必要がある。
        impulseモードでは、振幅と時間倍率が
        validate_impulse_parameters()の条件を満たさない場合も送出する。

    Notes
    -----
    設定値は変更しない。両モード共通でvalidate_time_config()により、
    時計・刻み設定・背景供給・ガス組成比も検査する。

    Examples
    --------
    >>> from streamer_drift_diffusion import SimulationConfig
    >>> validate_config(SimulationConfig(nr=2, nz=3, t_end=0.))
    """
    for name, minimum in (("nr", 2), ("nz", 2)):
        value = getattr(config, name)
        if not isinstance(value, Integral) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
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
    from .timestepping import validate_time_config
    validate_time_config(config)


def initialize_simulation(config=None):
    """共有格子・電極マスク・FiPy Poisson方程式・初期粒子分布を生成する。

    Parameters
    ----------
    config : SimulationConfig, optional
        Noneなら既定値。計算条件はここに集約。

    Returns
    -------
    SimulationState
        シミュレーションごとに独立した状態。ne_var・phi_varをFiPyの正本として確保し、
        ne・phiはその2次元ビューとする。座標等は1次元、
        密度・電位・中心電場等は(nr,nz)。半径面の電場・速度・移動度は
        (nr+1,nz)、軸方向面は(nr,nz+1)。electron_lossesは長さ3。

    Raises
    ------
    ValueError
        不正設定、または気体セルのない格子。

    Notes
    -----
    共有メッシュ上のFiPy Poisson方程式と境界電位を初期化する。
    電子密度のCellVariableはhasOld=True、電位はPoissonSolverと同じCellVariableを参照する。
    この2変数に対応する独立したNumPy状態配列は確保しない。
    初期phiと電場は0。時間発展時に総電位を直接求解する。
    初期の中性粒子・電荷中性Gaussian・背景供給は気体セルに限定する。
    電極内の全密度と累積流出電子数は0。gas_maskはbool、他はfloat64。

    Examples
    --------
    >>> from streamer_drift_diffusion import SimulationConfig
    >>> config = SimulationConfig(nr=2, nz=3, t_end=0.)
    >>> state = initialize_simulation(config)
    >>> state.ne.shape, state.vex.shape, state.vey.shape
    ((2, 3), (3, 3), (2, 4))
    """
    config = SimulationConfig() if config is None else config
    validate_config(config)
    arrays = initialize_grid(config)
    if not np.any(arrays['gas_mask']):
        raise ValueError("grid contains no gas cells; check electrode_gap, cathode_radius and delta")
    for name in ("er_face", "vex", "mobility_r"):
        arrays[name] = np.zeros((config.nr + 1, config.nz), dtype=np.float64)
    for name in ("ez_face", "vey", "mobility_z"):
        arrays[name] = np.zeros((config.nr, config.nz + 1), dtype=np.float64)
    arrays['poisson'] = PoissonSolver(arrays['grid'], arrays['gas_mask'], config)
    arrays['phi_var'] = arrays['poisson'].potential
    arrays['ne_var'] = CellVariable(mesh=arrays['grid'].mesh, value=0., hasOld=True)
    arrays['radial_boundary'] = arrays['poisson'].radial_boundary
    arrays['upper_boundary'] = arrays['poisson'].upper_boundary
    arrays['chemistry'] = None
    arrays['k1_o2'] = None
    if config.chemistry_table is not None:
        from .plasma_chemistry import load_reaction_table
        arrays['chemistry'] = load_reaction_table(config.chemistry_table, config)
        arrays['k1_o2'] = np.zeros(arrays['gas_mask'].shape)
    for field in fields(SimulationState):
        if field.name not in arrays:
            arrays[field.name] = np.zeros((config.nr, config.nz), dtype=np.float64)
    state = SimulationState(**arrays)
    r, z = state.rh[:, None], state.zh[None, :]
    # 初期電子・イオン　初期分布をGaussianで生成し、局所的な電荷中性を保つ
    gaussian = np.exp(-(r * r) / (2 * config.sr0 * config.sr0)
                      - ((z - config.z0) * (z - config.z0)) / (2 * config.sz0 * config.sz0))
    mol = molecular_density(config)
    state.n2[:] = config.nitrogen_fraction * mol * state.gas_mask
    state.o2[:] = config.oxygen_fraction * mol * state.gas_mask
    state.n2p[:] = config.nitrogen_fraction * config.nmax * gaussian * state.gas_mask
    state.o2p[:] = config.oxygen_fraction * config.nmax * gaussian * state.gas_mask
    state.ne_var.setValue((state.n2p + state.o2p).ravel(order='F'))
    calculate_transport_coefficients(state, config)
    return state
