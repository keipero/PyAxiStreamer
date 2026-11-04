"""双曲線電極、軸対称Poissonの並列RB-SOR、換算電場。"""
import math
import numpy as np

from ._acceleration import njit, prange
from .simulation_types import gap_length, molecular_density
from .voltage_waveforms import applied_voltage


@njit(cache=True, fastmath=False)
def hyperbolic_curve(vele, cr, b, r, z):
    """双曲線電極と接地平面による基礎電位を1点で評価する。

    Args:
        vele (float): 双曲線電極の印加電圧 [V]。
        cr (float): 電極先端の曲率半径 [m]。正の値を指定する。
        b (float): 接地平面から電極先端までのギャップ長 [m]。
            正の値を指定する。
        r (float): 評価点の半径方向座標 [m]。
        z (float): 評価点の軸方向座標 [m]。接地平面をz=0とする。

    Returns:
        float: 評価点の基礎電位 [V]。

    Notes:
        a = sqrt(cr * b)とし、(z/b)^2 - (r/a)^2 >= 1の電極領域では
        veleを返す。それ以外のz=0では0を返し、気体領域では
        双曲線電極の解析式を使う。空間電荷の寄与は含まない。
    """
    a = math.sqrt(cr * b)
    if (z / b) ** 2 - (r / a) ** 2 >= 1:
        return vele
    if z == 0:
        return 0.0
    theta_i = math.atan2(a, b)
    q = math.sqrt(b * b + a * a)
    distance = r * r + z * z - q * q
    a2_squared = (-distance + math.sqrt(distance ** 2 - 4 * (-r * r * q * q))) / 2
    a2 = math.sqrt(a2_squared)
    b2 = math.sqrt(q * q - a2 * a2)
    theta = math.atan2(a2, b2)
    return vele * math.log(1 / math.tan(theta / 2)) / math.log(1 / math.tan(theta_i / 2))


def discretization(rh, zh, p1, p2, p3, p4, p5, *, gas_mask=None,
                   radial_boundary=None, upper_boundary=None):
    """等間隔円筒格子の有限体積Poisson係数と、単位電圧の境界源を生成する。

    Args:
        rh, zh (np.ndarray): r=0、z=0から始まるセル中心座標 [m]。
        p1, p2, p3, p4, p5 (np.ndarray): 逆対角係数と上・内・外・下の係数の格納先。
        gas_mask (np.ndarray, optional): 気体=True、固定電位の電極=False。
        radial_boundary, upper_boundary (np.ndarray, optional): 最外周・上端の
            単位電圧時の電位。省略時は0。接地平面は常に0。

    Returns:
        np.ndarray: 1 V印加時の境界源 [V/m^2]。

    Notes:
        軸の面積は0。電極面はセル境界による階段状近似とし、Dirichlet面まで
        の距離は半セル幅。電極はp1=0として更新対象から除く。rhoは体積電荷
        密度/ε₀で、旧版の半径重みは使わない。座標は等間隔を前提とする。
    """
    nr, nz = p1.shape
    gas = np.ones((nr, nz), dtype=bool) if gas_mask is None else gas_mask
    radial = np.zeros(nz) if radial_boundary is None else radial_boundary
    upper = np.zeros(nr) if upper_boundary is None else upper_boundary
    dr, dz = 2 * rh[0], 2 * zh[0]
    source = np.zeros((nr, nz))
    for array in (p1, p2, p3, p4, p5):
        array[:] = 0.
    for i in range(nr):
        area = rh[i] * dr
        weights = (1/dz**2, (rh[i]-dr/2)/(area*dr),
                   (rh[i]+dr/2)/(area*dr), 1/dz**2)
        for j in range(nz):
            if not gas[i, j]:
                continue
            diagonal = 0.
            for weight, array, ni, nj in zip(weights, (p2, p3, p4, p5),
                                             (i, i-1, i+1, i), (j+1, j, j, j-1)):
                if weight == 0.:
                    continue
                if 0 <= ni < nr and 0 <= nj < nz and gas[ni, nj]:
                    array[i, j] = -weight
                    diagonal += weight
                else:
                    value = (radial[j] if ni == nr else upper[i] if nj == nz
                             else 0. if nj < 0 else 1.)
                    diagonal += 2 * weight
                    source[i, j] += 2 * weight * value
            p1[i, j] = 1 / diagonal
    return source


def initialize_potential(state, config):
    """境界係数を生成し、印加1 Vの離散ラプラス解を初期化時に1回求める。"""
    gap, radius = gap_length(config), config.cathode_radius
    radial = np.array([hyperbolic_curve(1., radius, gap, config.nr*config.delta, z)
                       for z in state.zh])
    upper = np.array([hyperbolic_curve(1., radius, gap, r, config.nz*config.delta)
                      for r in state.rh])
    state.radial_boundary[:] = radial
    state.upper_boundary[:] = upper
    source = discretization(state.rh, state.zh, state.p1, state.p2, state.p3, state.p4,
                            state.p5, gas_mask=state.gas_mask,
                            radial_boundary=radial, upper_boundary=upper)
    _base_potential(state.unit_potential, state.rh, state.zh, 1., radius, gap)
    poisson_calc(state.unit_potential, source, state.p1, state.p2, state.p3, state.p4,
                 state.p5, omega=config.poisson_omega, tolerance=config.poisson_tolerance,
                 verbose=False, max_iterations=config.poisson_max_iterations)


@njit(cache=True, parallel=True, fastmath=False)
def _poisson_sweeps(phi, rho, p1, p2, p3, p4, p5, omega, tolerance, max_phi, limit):
    """赤・黒の順に並列更新するRB-SORを、指定反復数まで実行する。

    Args:
        phi (np.ndarray): 初期推定電位と解の格納先 [V]。形状は(nr, nz)。
        rho (np.ndarray): 重み付きのPoisson電荷源。形状はphiと同じ。
        p1 (np.ndarray): 電位更新式全体に掛ける係数。
            p1～p5の形状はphiと同じ。
        p2 (np.ndarray): zが大きい側の隣接電位に掛ける係数。
        p3 (np.ndarray): rが小さい側の隣接電位に掛ける係数。
        p4 (np.ndarray): rが大きい側の隣接電位に掛ける係数。
        p5 (np.ndarray): zが小さい側の隣接電位に掛ける係数。
        omega (float): 緩和係数。1ならGauss–Seidel法。
        tolerance (float): 正規化した電位更新量の許容値（無次元）。
        max_phi (float): 収束判定の電位尺度 [V]。正の値を指定し、
            先行する走査で得た最大絶対電位を引き継ぐ。
        limit (int): この呼び出しで行う格子全体の走査回数の上限。

    Returns:
        tuple[int, float, bool]: 実行した走査回数、更新後のmax_phi、
            収束判定を満たしたかどうか。電位はphiにその場で反映する。

    Notes:
        (i+j)%2が0の赤セル、1の黒セルの順に更新して1反復とする。
        同色の行をprangeで分担し、色の間で同期する。
        Numba無効時も同じ更新式・順序をrangeで実行する。
        z上下端の外側電位は0、半径方向最外周の外側電位はそのセルの
        更新前の電位とする。軸では内側の隣接項を使わない。
        行別の最大更新量と最大絶対電位を集計し、1反復末尾の
        max|新電位-旧電位| / max_phiで収束を判定する。
        max_phiは反復間でリセットしない。非有限値は未収束として返す。
    """
    nr, nz = phi.shape
    row_delta = np.empty(nr)
    row_phi = np.empty(nr)
    for sweep in range(limit):
        for color in range(2):
            for i in prange(nr):
                delta = 0.0 if color == 0 else row_delta[i]
                scale = 0.0 if color == 0 else row_phi[i]
                for j in range((color + i % 2) % 2, nz, 2):
                    if p1[i, j] == 0.:
                        continue  # 電極セルの既知電位を保つ。
                    previous = phi[i, j]
                    upper = 0.0 if j == nz - 1 else phi[i, j + 1]
                    lower = 0.0 if j == 0 else phi[i, j - 1]
                    outer = previous if i == nr - 1 else phi[i + 1, j]
                    inner = 0.0 if i == 0 else phi[i - 1, j]
                    gs_value = p1[i, j] * (
                        rho[i, j] - p2[i, j] * upper
                        - (p3[i, j] * inner if i > 0 else 0.0)
                        - p4[i, j] * outer - p5[i, j] * lower
                    )
                    value = gs_value if omega == 1.0 else previous + omega * (gs_value - previous)
                    phi[i, j] = value
                    change = abs(value - previous) if math.isfinite(value) else math.inf
                    delta = max(delta, change)
                    scale = max(scale, abs(value))
                row_delta[i], row_phi[i] = delta, scale
        max_delta = 0.0
        for i in range(nr):
            max_delta = max(max_delta, row_delta[i])
            max_phi = max(max_phi, row_phi[i])
        if not math.isfinite(max_delta):
            return sweep + 1, max_phi, False
        if max_delta / max_phi <= tolerance:
            return sweep + 1, max_phi, True
    return limit, max_phi, False


def poisson_calc(phi, rho, p1, p2, p3, p4, p5, *, omega=1.5, tolerance=1e-6,
                 progress_every=20000, verbose=True, max_iterations=None):
    """並列RB-SOR法でPoisson方程式を解き、電位配列を更新する。

    Args:
        phi (np.ndarray): 初期推定電位と解の格納先 [V]。形状は(nr, nz)。
        rho (np.ndarray): 重み付きのPoisson電荷源。形状はphiと同じ。
        p1 (np.ndarray): 電位更新式全体に掛ける係数。
            p1～p5はdiscretization()で生成し、形状はphiと同じ。
        p2 (np.ndarray): zが大きい側の隣接電位に掛ける係数。
        p3 (np.ndarray): rが小さい側の隣接電位に掛ける係数。
        p4 (np.ndarray): rが大きい側の隣接電位に掛ける係数。
        p5 (np.ndarray): zが小さい側の隣接電位に掛ける係数。
        omega (float, optional): 0 < omega < 2を満たす緩和係数。
            1ならGauss–Seidel法。既定値は1.5。
        tolerance (float, optional): 正規化した電位更新量の許容値。
            正の値を指定する。既定値は1e-6。
        progress_every (int, optional): 未収束時に累積走査回数と電位尺度を
            表示する間隔。0以下なら表示しない。既定値は20000。
        verbose (bool, optional): 反復の進捗表示を有効にするかどうか。
            既定値はTrue。
        max_iterations (int, optional): 総走査回数の上限。Noneなら従来どおり
            無制限。非有限電位は上限の指定によらずエラーとする。

    Raises:
        ValueError: omegaが0 < omega < 2を満たさない場合。
            またはmax_iterationsが正の整数でない場合。
        RuntimeError: 指定した反復上限内に収束しない、または
            非有限電位が生じた場合。

    Returns:
        int: 収束までに実行した格子全体の走査回数。

    Notes:
        収束判定には方程式の残差ではなく、電位更新量を使う。
        電位尺度max_phiは1e-10 Vから始め、_poisson_sweeps()の
        呼び出し間でも保持する。max_iterations=Noneなら総上限はない。
        進捗表示を無効にした場合も、20000走査ずつ収束判定を伴う
        内部処理を呼び出し、収束した走査で終了する。
    """
    if not 0.0 < omega < 2.0:
        raise ValueError("omega must satisfy 0 < omega < 2")
    if max_iterations is not None and (not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1):
        raise ValueError("max_iterations must be a positive integer")
    loop, max_phi = 0, 1e-10
    # チャンク間でも max_phi を保持する。反復ごとにはリセットしない。
    chunk = progress_every if verbose and progress_every > 0 else 20000
    while True:
        limit = chunk if max_iterations is None else min(chunk, max_iterations - loop)
        count, max_phi, converged = _poisson_sweeps(phi, rho, p1, p2, p3, p4, p5, omega, tolerance, max_phi, limit)
        loop += count
        if not np.all(np.isfinite(phi)):
            raise RuntimeError("Poisson solve produced a nonfinite potential")
        if converged: # 収束したら，ループを抜ける
            return loop
        if max_iterations is not None and loop >= max_iterations:
            raise RuntimeError(f"Poisson solve did not converge in {loop} sweeps")
        if verbose and progress_every > 0:
            print(f"{loop:05d}  {max_phi:e}")


def calculate_charge_density(state, config):
    """粒子数密度から体積電荷源rho [V/m^2]を計算する。

    Args:
        state (SimulationState): ne、n2p、o2p、om、o2m [m^-3]と気体マスク。
        config (SimulationConfig): 電気素量と真空の誘電率。

    Returns:
        None: state.rhoをその場で更新する。電極内は0。

    Notes:
        rho = e/ε₀ * (n2p+o2p-om-o2m-ne)。半径重みは掛けない。
        円筒座標の幾何は有限体積Poisson係数側へ含める。"""
    positive = state.n2p + state.o2p
    negative = state.om + state.o2m + state.ne
    state.rho[:] = (config.elementary_charge / config.vacuum_permittivity) * (positive - negative)
    state.rho[~state.gas_mask] = 0.


@njit(cache=True, fastmath=False)
def _base_potential(lphi, rh, zh, voltage, cr, gap):
    """全セル中心で双曲線電極による基礎電位（ラプラス電位）を計算する。

    Args:
        lphi (np.ndarray): 基礎電位の格納先 [V]。形状は(nr, nz)。
        rh (np.ndarray): 半径方向のセル中心座標 [m]。形状は(nr,)。
        zh (np.ndarray): 軸方向のセル中心座標 [m]。形状は(nz,)。
        voltage (float): 評価時刻における印加電圧 [V]。
        cr (float): 電極先端の曲率半径 [m]。
        gap (float): ギャップ長 [m]。

    Returns:
        None: hyperbolic_curve()の評価値をlphiにその場で格納する。
    """
    for i in range(len(rh)):
        for j in range(len(zh)):
            lphi[i, j] = hyperbolic_curve(voltage, cr, gap, rh[i], zh[j])


def calculate_potential(state, config, *, verbose=True, time=0.0, waveform=None):
    """離散ラプラス解と空間電荷補正を合成し、全電位を求める。

    Args:
        state (SimulationState): 初期化済みのunit_potential、Poisson係数とrho。
        config (SimulationConfig): 電圧波形とPoisson反復の設定。
        verbose (bool, optional): 反復の進捗表示。既定True。
        time (float, optional): 電圧を評価する時刻 [s]。既定0。
        waveform (VoltageWaveform, optional): 読込済み波形。既定None。

    Returns:
        float: 現在の印加電圧 [V]。lphi、pphi、phiも更新する。

    Notes:
        lphiは初期化時に求めた離散単位電位×印加電圧。
        pphiは電極・接地平面・外周・上端で0、軸は対称条件とする。
        pphiの前step解を初期値とし、固定・適応の両モードで反復上限を使う。
        密度更新後は先にcalculate_charge_density()を呼ぶ。"""
    voltage = applied_voltage(config, time, waveform=waveform)
    # 初期化時に求めた離散ラプラス解を電圧倍する。
    state.lphi[:] = voltage * state.unit_potential
    state.pphi[~state.gas_mask] = 0.
    # 空間電荷による電位を並列RB-SORで計算。前stepの解を初期値に使う。
    poisson_calc(state.pphi, state.rho, state.p1, state.p2, state.p3, state.p4, state.p5,
                 omega=config.poisson_omega, tolerance=config.poisson_tolerance,
                 progress_every=config.poisson_progress_every, verbose=verbose,
                 max_iterations=config.poisson_max_iterations)
    # 全電位を合成する。phi = pphi + lphi
    state.phi[:] = state.pphi + state.lphi
    return voltage


def calculate_electric_field(state, config, *, voltage=None):
    """全セル面の法線電場と、反応・出力用のセル中心電場を求める。

    Args:
        state (SimulationState): 中心電位phi [V]、気体マスク、単位境界電位。
        config (SimulationConfig): 換算に用いるガス条件。
        voltage (float, optional): 現在の印加電圧 [V]。省略時はconfig.voltage。
            インパルス計算ではcalculate_potential()が返す実際の電圧を渡す。

    Returns:
        None: er_face (nr+1,nz)、ez_face (nr,nz+1)と、
            中心ex・ey・abs_e (nr,nz) [Td]を更新する。

    Notes:
        面法線電場はPoissonの有限体積係数と同じ距離・境界電位を使う。
        軸は法線電場0。電極・接地・上端・外周は半セル幅のDirichlet面。
        中心成分は対向する2面の算術平均で、中心abs_eはその2成分から求める。
        電極内部の面電場・中心電場は0、気体・電極間の法線電場は保持する。
        等間隔格子を対象とする。電場の単位は全てTd。"""
    voltage = config.voltage if voltage is None else voltage
    scale = molecular_density(config) * 1e-21
    grid, gas = state.grid, state.gas_mask
    boundary = np.full(grid.mesh.numberOfFaces, voltage, dtype=float)
    boundary[grid.face_ids[0][-1]] = voltage*state.radial_boundary
    boundary[grid.face_ids[1][:, 0]] = 0.
    boundary[grid.face_ids[1][:, -1]] = voltage*state.upper_boundary
    low_gas, high_gas = grid.adjacent(gas)
    low, high = grid.adjacent(state.phi)
    low, high = np.where(low_gas, low, boundary), np.where(high_gas, high, boundary)
    field = -(high-low)/(grid.distances(gas)*scale)
    field *= (low_gas | high_gas) & (grid.area > 0.)
    er, ez = state.er_face, state.ez_face
    er[:], ez[:] = grid.split_faces(field)
    state.ex[:] = .5 * (er[:-1] + er[1:]) * gas
    state.ey[:] = .5 * (ez[:, :-1] + ez[:, 1:]) * gas
    np.hypot(state.ex, state.ey, out=state.abs_e)
