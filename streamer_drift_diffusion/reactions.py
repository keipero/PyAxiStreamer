"""電子衝突電離、解離付着、三体付着、実際の刻みに比例する背景電離。"""
import numpy as np


def calculate_reaction_coefficients(state):
    """セル中心の換算電場から、3種類の反応係数を計算する。

    Parameters
    ----------
    state : SimulationState
        計算済みの換算電場の大きさabs_e [Td]、
        係数の格納先k1・k1_o2・k2・k3と任意の係数表。形状は(nr,nz)。
        abs_eは0以上の有限値であることを前提とする。

    Returns
    -------
    None
        N₂・O₂電離の係数をstate.k1・state.k1_o2 [m^3/s]、解離付着の係数を
        state.k2 [m^3/s]、三体付着の係数をstate.k3 [m^6/s]へ格納する。

    Raises
    ------
    RuntimeError
        係数表を使うとき、気体セルの換算電場が表の範囲外または不正。

    Notes
    -----
    表を指定した場合、二体反応係数は気体セルの中心電場から補間する。
    経験式ではk1をN₂とO₂の電離に共用する。
    三体付着はO₂を第三体とする経験式を使う。BOLOS輸送では、
    0 Tdでの値0と表の最小正電場点での経験値を結び、低電場での発散を避ける。
    経験式の速度は中心の電場強度から計算し、abs_eが0では係数を0とする。
    入力の電場・速度と粒子密度は変更せず、係数配列をその場で更新する。

    Examples
    --------
    >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
    >>> config = SimulationConfig(nr=2, nz=3, voltage=0., nmax=0.,
    ...                           background_ionization=0., t_end=0.)
    >>> state = initialize_simulation(config)
    >>> calculate_reaction_coefficients(state)
    >>> float(state.k1.max()), float(state.k2.max()), float(state.k3.max())
    (0.0, 0.0, 0.0)
    """
    table = state.chemistry
    if table is not None:
        rates = table.sample(state.abs_e[state.gas_mask])
        for target, rate in zip((state.k1, state.k1_o2, state.k2), rates):
            target[:] = 0.
            target[state.gas_mask] = rate
    nonzero = (state.abs_e != 0.) & state.gas_mask
    field = state.abs_e[nonzero]
    attachment_scale = (5.0 * 5.0) * 1.6e-47
    speed_scale = 3.74e24 * 1e-21
    state.k3[:] = 0.0
    if table is not None and table.swarm is not None:
        reference_field = np.maximum(field, table.field_td[1])
        rate_at_reference = attachment_scale * speed_scale * reference_field**(-.35)
        state.k3[nonzero] = rate_at_reference * (field/reference_field)
        return
    speed = speed_scale * field**.75
    if table is None:
        state.k1[:] = 0.
        state.k2[:] = 0.
        state.k1[nonzero] = 1.4e-20 * np.exp(-(660.0 / field)) * speed
        state.k2[nonzero] = 5.0 * 6.0e-23 * np.exp(-(100 / field)) * speed
    state.k3[nonzero] = attachment_scale * field**(-1.1) * speed


def apply_reactions(state, config, *, dt):
    """電子衝突反応と背景電離を適用し、各粒子種の密度を更新する。

    Parameters
    ----------
    state : SimulationState
        電子ne、中性粒子n2・o2・o、正イオンn2p・o2p、
        負イオンom・o2mの数密度 [m^-3] と、計算済みの反応係数
        k1・k1_o2・k2 [m^3/s]、k3 [m^6/s] を保持する状態。
        各配列の形状は(nr, nz)。密度配列をその場で更新する。
    config : SimulationConfig
        背景供給の基準刻みdt [s]、その基準幅に対応する密度加算量
        background_ionization [m^-3]、背景正イオンの分配に使う
        nitrogen_fraction・oxygen_fractionを保持する計算設定。
    dt : float
        実際の更新幅 [s]。両モードで必ず指定し、背景加算量を
        background_ionization * dt / config.dtへ換算する。

    Returns
    -------
    None
        反応と背景電離を適用した密度をstateへ反映する。

    Notes
    -----
    Canteraの反応機構を含む係数表では、速度式の次数と化学量論から
    各種の増分を求める。以下は同梱の4反応と従来の係数表の更新式。

    以下の反応量を、全てこの関数を呼び出した時点の密度から
    陽的に計算する。r1n・r1o・r2・r3の単位はいずれも[m^-3]。
    r1n = dt * k1 * n2 * ne：N₂の電子衝突電離。
    r1o = dt * k1_o2 * o2 * ne：O₂の電子衝突電離。
    r2 = dt * k2 * o2 * ne：解離付着によるO⁻の生成。
    r3 = dt * k3 * o2^2 * ne：三体付着によるO₂⁻の生成。

    電離では電子と対応する正イオンを増やし、中性粒子を減らす。
    付着では電子とO₂を減らし、対応する負イオンを増やす。
    三体付着の第三体O₂は反応後も残るため、O₂の減少量はr3とする。
    解離付着ではO⁻と中性Oを同数生成する。

    背景電離は実際のdtに比例する密度を電子へ加え、正イオンへ組成比で
    分配する。中性粒子の減少は適用しない。両組成比の和が1の場合、
    背景項による電子と正イオンの生成電荷は相殺する。

    共通のadvance_density()から、輸送後の電子密度に適用する。
    反応係数の再計算、電荷源・電位・電場の更新、
    負密度の補正や反応量の制限は行わない。
    電子密度はne_var.setValue()で正本を更新し、FiPyの派生式のキャッシュにも変更を通知する。
    他の密度はNumPy配列をその場で更新する。

    Examples
    --------
    >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
    >>> config = SimulationConfig(nr=2, nz=3, voltage=0., nmax=0.,
    ...                           background_ionization=0., t_end=0.)
    >>> state = initialize_simulation(config)
    >>> from dataclasses import replace
    >>> config = replace(config, background_ionization=1e12)
    >>> apply_reactions(state, config, dt=config.dt/2)
    >>> float(state.ne[0, 0])
    500000000000.0
    """
    background = config.background_ionization * (dt/config.dt) * state.gas_mask
    if state.chemistry is not None and state.chemistry.mechanism is not None:
        increments = state.chemistry.mechanism.increments(state, dt)
    else:
        r1n = dt * state.k1 * state.n2 * state.ne
        k_o2 = state.k1 if state.k1_o2 is None else state.k1_o2
        r1o = dt * k_o2 * state.o2 * state.ne
        r2 = dt * state.k2 * state.o2 * state.ne
        r3 = dt * state.k3 * state.o2 * state.o2 * state.ne
        increments = r1n + r1o - r2 - r3, -r1n, -r1o - r2 - r3, r1n, r1o, r2, r3, r2
    ne, n2, o2, n2p, o2p, om, o2m, o = increments
    state.ne_var.setValue((state.ne + (ne + background)).ravel(order='F'))
    state.n2[:] += n2
    state.o2[:] += o2
    state.n2p[:] += n2p + config.nitrogen_fraction * background
    state.o2p[:] += o2p + config.oxygen_fraction * background
    state.om[:] += om
    state.o2m[:] += o2m
    state.o[:] += o
