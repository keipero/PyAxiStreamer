"""電子ドリフト速度と電子拡散係数。係数の格納先は全てstateに集約する。"""
import numpy as np

from .simulation_types import molecular_density


def electron_drift(component, field):
    """換算電場の1成分から、その方向の電子ドリフト速度を求める。

    Parameters
    ----------
    component : float
        半径方向または軸方向の換算電場成分 [Td]。
        正負の符号を持ち、state.exまたはstate.eyに相当する。
    field : float
        換算電場の大きさ [Td]。state.abs_eに相当する
        0以上の有限値を指定する。

    Returns
    -------
    float
        指定方向の電子ドリフト速度 [m/s]。
        fieldが0の場合は0.0、それ以外は元コードの式
        -3.74e24 * (component * 1e-21) * field**(-0.25)を返す。

    Notes
    -----
    負号は電子が電場と反対方向へドリフトすることを表す。
    componentの1e-21倍は、Tdから換算電場のSI単位 [V m^2] への
    換算であり、電場 [V/m] への換算ではない。
    引数の妥当性はこの関数では検査しない。
    通常の時間発展では配列版のcalculate_transport_coefficients()を使う。

    Examples
    --------
    >>> electron_drift(0., 0.)
    0.0
    >>> electron_drift(-70., 70.) > 0.
    True
    """
    return 0.0 if field == 0 else -3.74e24 * (component * 1e-21) * field ** (-0.25)


def electron_mobility(field, config):
    """換算電場強度から、現行ドリフト式の電子移動度を求める。

    Parameters
    ----------
    field : ndarray
        換算電場強度 [Td]。0以上の有限値を前提とする。
    config : SimulationConfig
        計算条件。単位と各設定の既定値はSimulationConfigを参照。

    Returns
    -------
    ndarray
        fieldと同じ形状の移動度 [m²/(V s)]。field=0では0。

    Notes
    -----
    F>0でmu=3.74e24*F**(-1/4)/N。Nは分子数密度 [m⁻³]。
    低電場の特異性は正則化しない。引数の妥当性はここでは検査しない。

    Examples
    --------
    >>> import numpy as np
    >>> from streamer_drift_diffusion import SimulationConfig
    >>> electron_mobility(np.array([0., 100.]), SimulationConfig()).shape
    (2,)
    >>> float(electron_mobility(np.array([0.]), SimulationConfig())[0])
    0.0
    """
    mobility = np.zeros_like(field, dtype=np.float64)
    np.power(field, -.25, out=mobility, where=field > 0.)
    return mobility * (3.74e24 / molecular_density(config))


def _tangential_faces(values, gas, axis, interior):
    """中心の接線電場を指定方向の全面へ補間する。

    Parameters
    ----------
    values : ndarray, shape (nr, nz)
        面の法線方向と直交する中心電場成分 [Td]。
    gas : ndarray of bool, shape (nr, nz)
        気体True、電極False。
    axis : int
        面の法線方向。0はr面、1はz面。
    interior : ndarray of bool
        共有格子が判定した気体間の全面マスク。axis=0では(nr+1,nz)、1では(nr,nz+1)。

    Returns
    -------
    ndarray
        全指定方向面の接線電場 [Td]。

    Notes
    -----
    気体間は隣接中心の平均、金属面は0。軸の接線成分は偶対称、接地面は0。
    外周・上端は気体2点から外挿し、隣が金属なら1点値を用いる。
    法線成分はこの補間で求めず、電位と境界値から直接計算する。

    Examples
    --------
    >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
    >>> config = SimulationConfig(nr=2, nz=3, voltage=0., nmax=0.,
    ...                           background_ionization=0., t_end=0.)
    >>> state = initialize_simulation(config)
    >>> interior_r, interior_z = state.grid.split_faces(state.grid.interior(state.gas_mask))
    >>> result = _tangential_faces(state.ey, state.gas_mask, 0, interior_r)
    >>> result.shape
    (3, 3)
    """
    values, gas = np.moveaxis(values, axis, 0), np.moveaxis(gas, axis, 0)
    interior = np.moveaxis(interior, axis, 0)
    result = np.zeros((values.shape[0] + 1, *values.shape[1:]))
    result[1:-1] = np.where(interior[1:-1], .5*(values[:-1]+values[1:]), 0.)
    if axis == 0:
        result[0] = values[0]  # r=0で軸方向成分は偶対称。
    result[-1] = np.where(gas[-2], 1.5*values[-1]-.5*values[-2], values[-1]) * gas[-1]
    return np.moveaxis(result, 0, axis)


def calculate_transport_coefficients(state, config):
    """各面の同じ位置で電場強度・移動度・法線速度を求め、中心Dも更新する。

    Parameters
    ----------
    state : SimulationState
        中心ex/eyと全法線面er_face/ez_face [Td]。
        vex・mobility_rは(nr+1,nz)、vey・mobility_zは(nr,nz+1)。
        任意の係数表からμNとDNを参照する。
    config : SimulationConfig
        分子数密度の計算条件と、従来モードの一定拡散係数 [m²/s]。

    Returns
    -------
    None
        面の移動度 [m^2/(V s)]・速度 [m/s]と中心deを更新する。

    Raises
    ------
    RuntimeError
        BOLOS表を補間するE/Nが範囲外または不正。

    Notes
    -----
    各面で法線成分と補間した接線成分のhypotからE/Nの大きさを求める。
    BOLOS表では面のμNと中心のDNをNで割り、面速度は法線方向の-μEとする。
    Dは全方向共通のスカラー。軸の半径流速・移動度、電極内部の係数は0。
    気体に接する吸収面にはBOLOS表の移動度を使う。
    輸送列のない場合はドリフト式と一定拡散係数を使う。

    Examples
    --------
    >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
    >>> config = SimulationConfig(nr=2, nz=3, voltage=0., nmax=0.,
    ...                           background_ionization=0., t_end=0.)
    >>> state = initialize_simulation(config)
    >>> calculate_transport_coefficients(state, config)
    >>> float(abs(state.vex).max()), float(abs(state.vey).max()), float(state.de.max())
    (0.0, 0.0, 0.0)
    """
    interior = state.grid.split_faces(state.grid.interior(state.gas_mask))
    radial = np.hypot(state.er_face, _tangential_faces(state.ey, state.gas_mask, 0, interior[0]))
    axial = np.hypot(_tangential_faces(state.ex, state.gas_mask, 1, interior[1]), state.ez_face)
    density = molecular_density(config)
    table = state.chemistry
    if table is None or table.swarm is None:
        state.mobility_r[:] = electron_mobility(radial, config)
        state.mobility_z[:] = electron_mobility(axial, config)
        state.de[:] = config.electron_diffusion * state.gas_mask
    else:
        mobility_n, diffusion_n = table.swarm.T
        low, high = state.grid.adjacent(state.gas_mask)
        gas_r, gas_z = state.grid.split_faces((low | high) & (state.grid.area > 0.))
        mu_r = table.sample(radial[gas_r], column=mobility_n) / density
        mu_z = table.sample(axial[gas_z], column=mobility_n) / density
        diffusion = table.sample(state.abs_e[state.gas_mask], column=diffusion_n) / density
        state.mobility_r.fill(0.)
        state.mobility_z.fill(0.)
        state.de.fill(0.)
        state.mobility_r[gas_r] = mu_r
        state.mobility_z[gas_z] = mu_z
        state.de[state.gas_mask] = diffusion
    state.mobility_r[0] = 0.
    scale = density * 1e-21
    state.vex[:] = -state.mobility_r * state.er_face * scale
    state.vey[:] = -state.mobility_z * state.ez_face * scale
