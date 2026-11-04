"""電子衝突電離、解離付着、三体付着、毎更新の背景電離。"""
import numpy as np


def calculate_reaction_coefficients(state):
    """セル中心の換算電場から、3種類の反応係数を計算する。

    Args:
        state (SimulationState): 計算済みの換算電場の大きさabs_e [Td]、
            係数の格納先k1・k2・k3を保持する状態。形状は(nr,nz)。
            abs_eは0以上の有限値であることを前提とする。

    Returns:
        None: 電子衝突電離の係数をstate.k1 [m^3/s]、解離付着の係数を
            state.k2 [m^3/s]、三体付着の係数をstate.k3 [m^6/s]へ格納する。

    Notes:
        元コードの速度式に中心の電場強度を代入し、
        speed=3.74e24*1e-21*abs_e**(3/4)とする。面速度は混ぜない。
        k1はN₂・O₂の電離に共通で、
        k2はO₂への解離付着、k3はO₂を第三体とする三体付着に対応する。
        abs_eが0のセルでは各係数を0とし、負のべき乗を評価しない。
        入力の電場・速度と粒子密度は変更せず、係数配列をその場で更新する。
    """
    nonzero = state.abs_e != 0.0
    field = state.abs_e[nonzero]
    speed = 3.74e24 * 1e-21 * field**.75
    state.k1[:] = 0.0
    state.k2[:] = 0.0
    state.k3[:] = 0.0
    state.k1[nonzero] = 1.4e-20 * np.exp(-(660.0 / field)) * speed
    state.k2[nonzero] = 5.0 * 6.0e-23 * np.exp(-(100 / field)) * speed
    state.k3[nonzero] = (5.0 * 5.0) * 1.6e-47 * field ** (-1.1) * speed


def apply_reactions(state, config, *, dt=None):
    """電子衝突反応と背景電離を適用し、各粒子種の密度を更新する。

    Args:
        state (SimulationState): 電子ne、中性粒子n2・o2、正イオンn2p・o2p、
            負イオンom・o2mの数密度 [m^-3] と、計算済みの反応係数
            k1・k2 [m^3/s]、k3 [m^6/s] を保持する状態。
            各配列の形状は(nr, nz)。密度配列をその場で更新する。
        config (SimulationConfig): 時間刻みdt [s]、毎更新の背景密度加算量
            background_ionization [m^-3]、背景正イオンの分配に使う
            nitrogen_fraction・oxygen_fractionを保持する計算設定。
        dt (float, optional): 実際の更新幅 [s]。指定時は背景加算量を
            background_ionization * dt / config.dtへ換算する。
            Noneの場合は従来の固定刻み・毎更新加算を使用する。

    Returns:
        None: 反応と背景電離を適用した密度をstateへ反映する。

    Notes:
        以下の反応量を、全てこの関数を呼び出した時点の密度から
        陽的に計算する。r1n・r1o・r2・r3の単位はいずれも[m^-3]。
        r1n = dt * k1 * n2 * ne：N₂の電子衝突電離。
        r1o = dt * k1 * o2 * ne：O₂の電子衝突電離。
        r2 = dt * k2 * o2 * ne：解離付着によるO⁻の生成。
        r3 = dt * k3 * o2^2 * ne：三体付着によるO₂⁻の生成。

        電離では電子と対応する正イオンを増やし、中性粒子を減らす。
        付着では電子とO₂を減らし、対応する負イオンを増やす。
        三体付着の第三体O₂は反応後も残るため、O₂の減少量はr3とする。
        解離付着で生成する中性Oは、このモデルでは追跡しない。

        背景電離は各セルの電子密度へbackground_ionizationを加え、
        正イオンへ両組成比で分配して加える。dt省略時の背景項にはdtを掛けず、
        中性粒子の減少も適用しない。両組成比の和が1の場合、
        背景項による電子と正イオンの生成電荷は相殺する。

        通常のadvance_step()では、移流後の電子密度に適用する。
        反応係数の再計算、電荷源・電位・電場の更新、
        負密度の補正や反応量の制限は行わない。
    """
    background = config.background_ionization * state.gas_mask
    if dt is None:
        dt = config.dt
    else:
        background *= dt / config.dt
    r1n = dt * state.k1 * state.n2 * state.ne
    r1o = dt * state.k1 * state.o2 * state.ne
    r2 = dt * state.k2 * state.o2 * state.ne
    r3 = dt * state.k3 * state.o2 * state.o2 * state.ne
    state.ne[:] += r1n + r1o - r2 - r3 + background
    state.n2[:] += -r1n
    state.o2[:] += -r1o - r2 - r3
    state.n2p[:] += r1n + config.nitrogen_fraction * background
    state.o2p[:] += r1o + config.oxygen_fraction * background
    state.om[:] += r2
    state.o2m[:] += r3
