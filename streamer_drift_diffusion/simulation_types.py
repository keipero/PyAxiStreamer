"""計算設定と状態。配列添字は [半径 i, 軸方向 j]、単位は SI（電場は Td）。"""
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from .grid import CylindricalGrid
from .kinetics import DENSITY_NAMES, DRY_AIR_SPECIES_FIELDS

if TYPE_CHECKING:
    from .electrostatics import PoissonSolver
    from .plasma_chemistry import ReactionTable
    from .photoionization import PhotoionizationSolver

# プロジェクトルートの絶対パス
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class SimulationConfig:
    """格子、物理定数、電圧、時間刻み・出力をまとめた不変の設定。

    Parameters
    ----------
    nr : int, optional
        半径方向の分割数
    nz : int, optional
        軸方向の分割数。既定の上端は1.5 mm、先端は1 mm。
    delta : float, optional
        格子間隔 [m]
    temperature : float, optional
        温度 [K]
    avogadro : float, optional
        アボガドロ数 [mol^-1]
    gas_constant : float, optional
        気体定数 [L·atm/(K·mol)]
    elementary_charge : float, optional
        電子の電荷 [C]
    vacuum_permittivity : float, optional
        真空の誘電率 [F/m]
    oxygen_fraction : float, optional
        酸素分率
    nitrogen_fraction : float, optional
        窒素分率
    cathode_radius : float, optional
        カソード半径 [m]
    voltage : float, optional
        DCモードの電圧 [V]
    voltage_mode : {"dc", "impulse"}, optional
        電圧モード。既定"dc"。
    impulse_peak_voltage : float, optional
        インパルス電圧のピーク値 [V]
    impulse_time_scale : float, optional
        インパルス電圧の時間倍率（無次元）
    voltage_waveform_file : str, optional
        インパルス電圧波形ファイルのパス
    dt : float, optional
        時間刻み [s]
    nmax : float, optional
        最大電子密度 [m^-3]
    initial_uniform_density : float, optional
        気体全域に一様に置く準中性の初期電子密度 [m^-3]。正イオンはN₂⁺・O₂⁺に組成比で分配し、
        Gaussian種に加える。既定0（Komuroらは5e9 m^-3 = 5e3 cm^-3）。
    z0 : float, optional
        初期電子密度の中心位置 [m]
    sr0 : float, optional
        初期電子密度の半径方向の広がり [m]
    sz0 : float, optional
        初期電子密度の軸方向の広がり [m]
    background_ionization : float, optional
        基準幅dtに対応する背景の密度加算量 [m⁻³]。実際はactual_dt/dtを掛けて供給する。
    poisson_tolerance : float, optional
        max|Aφ-b|/max|b| の許容値。旧SOR更新量とは異なる。
    adaptive_timestep : bool, optional
        共通ループで刻み幅の選択・再試行だけを切り替える。
    t_end : float, optional
        両モード共通の終了時刻 [s]
    dt_max : float or None, optional
        適応刻みの上限 [s]。Noneならdt。固定刻みの基本幅には使用しない。
    dt_min : float, optional
        適応候補刻みの停止下限 [s]。イベントへの安全な端数短縮は下限未満も許す。
    dt_safety_factor : float, optional
        各安定条件へ掛ける安全係数
    dt_cfl_number : float, optional
        セルの流出率に対するCFL数
    dt_chemistry_fraction : float, optional
        反応の生成・消費を制限する割合
    dt_growth_factor : float, optional
        受理した刻みからの増加倍率の上限
    max_step_retries : int, optional
        密度更新の縮小・再試行回数の上限
    max_steps : int, optional
        両モード共通の受理step数の停止上限
    output_interval : float, optional
        出力の実時間間隔 [s]
    electrode_gap : float, optional
        接地平面z=0から電極先端までの距離[m]。領域寸法とは独立。
    electron_diffusion : float, optional
        従来モードの一定電子拡散係数 [m²/s]。0は拡散なし。
        BOLOS輸送列のある表を使う場合は既定値0を指定する。
    chemistry_table : str or pathlib.Path or None, optional
        Py3BOLOSとCanteraで生成した電子輸送・反応係数表。Noneなら従来の経験式。

    photoionization_efficiency : float, optional
        発光源の無次元係数。消光補正前の値で、範囲は(0, 1]、既定0.075。
    photoionization_workers : int, optional
        Helmholtz求解のスレッド数。1～3、既定3。
    positive_ion_mobility, negative_ion_mobility : float, optional
        正・負イオンの移動度 [m²/(V s)]。0（既定）ならその符号のイオンは静止。
        正なら全ての正・負イオン種を一次風上で面の電場に沿って移動させる。
    flux_neumann_boundaries : bool, optional
        上端・外周の電位と電子流束を法線勾配0とする。既定Falseは解析電位と吸収境界。
    laplace_flux_boundaries : {"none", "outer", "outer_and_top"}, optional
        指定した外端面の法線電場を双曲線電極の解析Laplace電場に固定する。
        そこでは空間電荷による電位が同次Neumann（Komuroらの分割に相当）。
        "outer"は半径方向外周だけ、"outer_and_top"は上端も。既定"none"は解析電位のDirichlet。
        電子・イオンは吸収境界のまま。flux_neumann_boundariesとは併用できない。
    electrode_contact_radius : float, optional
        "outer_and_top"で、上端のうち電極電位（解析値）に固定する軸付近の半径 [m]。
        Noneなら上端の1セル上での双曲線電極の半径。これより外側の上端面だけ法線電場を固定する。
    reaction_buffer_height : float, optional
        上端から下向きにこの厚さ [m] の層で、反応・光電離（発光と吸収）・背景電離を止める。
        輸送とイオン移動は続ける。電極胴体が上端を横切る角での人為的な放電を抑える。既定0。

    Notes
    -----
    単位はSI、換算電場だけTdを用いる。フィールドの再代入は禁止。
    値の検査はvalidate_config()およびinitialize_simulation()で行い、dataclass生成だけでは検査しない。
    固定／適応ともt_end、output_interval、max_stepsを共用し、刻み選択・再試行方針だけを切り替える。
    出力先・出力有効フラグは設定に含めず、run_simulation()へ渡す。

    Examples
    --------
    >>> from dataclasses import replace
    >>> config = SimulationConfig(nr=8, nz=12, t_end=3e-12)
    >>> adaptive = replace(config, adaptive_timestep=True)
    >>> config.adaptive_timestep, adaptive.adaptive_timestep
    (False, True)
    """
    nr: int = 200 # 半径方向の分割数
    nz: int = 150 # 軸方向の分割数。既定の上端は1.5 mm、先端は1 mm。
    delta: float = 10e-6 # 格子間隔 [m]
    temperature: float = 300.0 # 温度 [K]
    avogadro: float = 6.022141e23 # アボガドロ数 [mol^-1]
    gas_constant: float = 0.082057366080960 # 気体定数 [L·atm/(K·mol)]
    elementary_charge: float = 1.602176462e-19 # 電子の電荷 [C]
    vacuum_permittivity: float = 8.85e-12 # 真空の誘電率 [F/m]
    oxygen_fraction: float = 0.2 # 酸素分率
    nitrogen_fraction: float = 0.8 # 窒素分率
    cathode_radius: float = 300e-6 # カソード半径 [m]
    voltage: float = 4000.0 # DCモードの電圧 [V]
    voltage_mode: Literal["dc", "impulse"] = "dc" # 電圧モード（"dc" または "impulse"）
    impulse_peak_voltage: float = 5000.0 # インパルス電圧のピーク値 [V]
    impulse_time_scale: float = 1.0 # インパルス電圧の時間倍率（無次元）
    voltage_waveform_file: str = str(Path(__file__).resolve().parent / "data" / "impulse_waveform_V50_tr5_tw6.txt") # インパルス電圧波形ファイルのパス
    dt: float = 1.0e-12 # 時間刻み [s]
    nmax: float = 1e19 # 最大電子密度 [m^-3]
    initial_uniform_density: float = 0.0  # 全域一様の準中性初期電子密度 [m^-3]
    z0: float = 900e-6 # 初期電子密度の中心位置 [m]
    sr0: float = 100e-6 # 初期電子密度の半径方向の広がり [m]
    sz0: float = 100e-6 # 初期電子密度の軸方向の広がり [m]
    background_ionization: float = 1e13 # 基準幅dtの密度加算量。実際はactual_dt/dtを掛ける。
    poisson_tolerance: float = 1e-10 # max|Aφ-b|/max|b| の許容値。旧SOR更新量とは異なる。
    adaptive_timestep: bool = False # 共通ループで刻み幅の選択・再試行だけを切り替える。
    t_end: float = 5.001e-9 # 両モード共通の終了時刻 [s]
    dt_max: float | None = None # 適応刻みの上限 [s]。Noneならdtを使用。
    dt_min: float = 1e-18 # 安定性のためにこれ未満が必要なら停止 [s]
    dt_safety_factor: float = 0.7 # 各安定条件へ掛ける安全係数
    dt_cfl_number: float = 0.8 # セルの流出率に対するCFL数
    dt_chemistry_fraction: float = 0.1 # 反応の生成・消費を制限する割合
    dt_growth_factor: float = 1.2 # 受理した刻みからの増加倍率の上限
    max_step_retries: int = 12 # 密度更新の縮小・再試行回数の上限
    max_steps: int = 1000000 # 両モード共通の受理step数の停止上限
    output_interval: float = 1e-10 # 出力の実時間間隔 [s]
    electrode_gap: float = 1e-3 # 接地平面z=0から電極先端までの距離[m]。領域寸法とは独立。
    electron_diffusion: float = 0.0 # 従来モード用の一定電子拡散係数 [m²/s]。
    chemistry_table: str | Path | None = None
    photoionization_efficiency: float = 0.075
    photoionization_workers: int = 3
    positive_ion_mobility: float = 0.0  # 正イオンの移動度 [m²/(V s)]。0なら静止
    negative_ion_mobility: float = 0.0  # 負イオンの移動度 [m²/(V s)]。0なら静止
    flux_neumann_boundaries: bool = False  # 添付図の外周電位・電子流束境界条件
    laplace_flux_boundaries: Literal["none", "outer", "outer_and_top"] = "none"  # 解析Laplace電場で法線電場を固定する外端
    electrode_contact_radius: float | None = None  # 上端で電極電位に固定する半径 [m]。Noneは自動
    reaction_buffer_height: float = 0.0  # 上端の反応停止層の厚さ [m]


def molecular_density(config):
    """理想気体の状態方程式から1 atmの分子数密度を求める。

    Parameters
    ----------
    config : SimulationConfig
        avogadro [mol⁻¹]、gas_constant [L atm/(K mol)]、temperature [K]を用いる。

    Returns
    -------
    float
        1 atmの分子数密度 [m⁻³]。

    Notes
    -----
    N = avogadro * 1000 / (gas_constant * temperature)。圧力は1 atmに固定し、
    1000はLからm³への体積換算に由来する。引数の妥当性はここでは検査しない。

    Examples
    --------
    >>> from streamer_drift_diffusion import SimulationConfig
    >>> 2e25 < molecular_density(SimulationConfig()) < 3e25
    True
    """
    return config.avogadro * 1000.0 / (config.gas_constant * config.temperature)


def gap_length(config):
    """接地平面から電極先端までの距離を返す。

    Parameters
    ----------
    config : SimulationConfig
        電極先端位置electrode_gap [m]を保持する設定。

    Returns
    -------
    float
        領域寸法とは独立した電極先端の高さ [m]。

    Examples
    --------
    >>> from streamer_drift_diffusion import SimulationConfig
    >>> gap_length(SimulationConfig(electrode_gap=1e-3))
    0.001
    """
    return config.electrode_gap


# frozen=Trueにより，frozen dataclass となり，フィールドの再代入が禁止される
@dataclass(frozen=True)
class SimulationState:
    """共有格子・ソルバーと、時間発展する物理量配列を保持する。

    Parameters
    ----------
    poisson : PoissonSolver
        総電位を直接求解するPoissonSolver。
    grid : CylindricalGrid
        全処理が共有する円筒メッシュ・面情報。
    r : ndarray of float64, shape (nr,)
        半径方向の各セルの下側境界 [m]。最外側境界は含まない
    z : ndarray of float64, shape (nz,)
        軸方向の各セルの下側境界 [m]。最上側境界は含まない
    rh : ndarray of float64, shape (nr,)
        半径方向のセル中心座標 [m]
    zh : ndarray of float64, shape (nz,)
        軸方向のセル中心座標 [m]
    dr : ndarray of float64, shape (nr,)
        半径方向のセル幅 [m]
    dz : ndarray of float64, shape (nz,)
        軸方向のセル幅 [m]
    volume : ndarray of float64, shape (nr, nz)
        セル体積係数 [m^3]（共通係数2πを省略）
    sr : ndarray of float64, shape (nr, nz)
        半径方向の下側セル面の面積係数 [m^2]（2πを省略）
    sz : ndarray of float64, shape (nr, nz)
        軸方向の下側セル面の面積係数 [m^2]（2πを省略）
    phi : ndarray of float64, shape (nr, nz)
        総電位 [V]。poisson.potentialと同じ配列。
    ex : ndarray of float64, shape (nr, nz)
        セル中心の半径方向換算電場E_r/N [Td]（名前のxはr方向）
    ey : ndarray of float64, shape (nr, nz)
        セル中心の軸方向換算電場E_z/N [Td]（名前のyはz方向）
    abs_e : ndarray of float64, shape (nr, nz)
        セル中心の換算電場の大きさ|E|/N [Td]。反応に使用
    n2 : ndarray of float64, shape (nr, nz)
        中性窒素分子N₂の数密度 [m^-3]
    o2 : ndarray of float64, shape (nr, nz)
        中性酸素分子O₂の数密度 [m^-3]
    n2p : ndarray of float64, shape (nr, nz)
        窒素分子正イオンN₂⁺の数密度 [m^-3]
    o2p : ndarray of float64, shape (nr, nz)
        酸素分子正イオンO₂⁺の数密度 [m^-3]
    om : ndarray of float64, shape (nr, nz)
        酸素原子負イオンO⁻の数密度 [m^-3]
    o2m : ndarray of float64, shape (nr, nz)
        酸素分子負イオンO₂⁻の数密度 [m^-3]
    ne : ndarray of float64, shape (nr, nz)
        電子密度 [m⁻³]。
    rho : ndarray of float64, shape (nr, nz)
        Poisson電荷源 [V/m^2]。正味電荷密度/ε₀（半径重みなし）
    vex : ndarray of float64, shape (nr+1, nz)
        全半径面の電子ドリフト速度 [m/s]。形状(nr+1,nz)
    vey : ndarray of float64, shape (nr, nz+1)
        全軸方向面の電子ドリフト速度 [m/s]。形状(nr,nz+1)
    k1 : ndarray of float64, shape (nr, nz)
        N₂の電子衝突電離の反応係数 [m^3/s]。
    k1_o2 : ndarray of float64 or None
        O₂の電子衝突電離の反応係数 [m^3/s]。係数表モードでは形状(nr,nz)。
        Noneなら経験式のk1をN₂とO₂に共通で使う。
    k2 : ndarray of float64, shape (nr, nz)
        O₂への解離付着によってO⁻を生成する反応係数 [m^3/s]
    k3 : ndarray of float64, shape (nr, nz)
        O₂を第三体とする三体付着でO₂⁻を生成する反応係数 [m^6/s]
    o : ndarray of float64, shape (nr, nz)
        解離付着で生じる中性酸素原子Oの数密度 [m⁻³]。
    chemistry : ReactionTable or None
        二体電子反応係数表。Noneなら従来の経験式を使う。
    gas_mask : ndarray of bool, shape (nr, nz)
        気体セル=True、電極セル=Falseのbool配列
    electron_losses : ndarray of float64, shape (3,)
        累積流出電子数 [電極、接地平面、上端＋外周]（2πを含む）
    de : ndarray of float64, shape (nr, nz)
        気体セルの電子拡散係数 [m^2/s]。BOLOS表または一定係数を使用し、電極内は0。
    er_face : ndarray of float64, shape (nr+1, nz)
        全半径面の法線方向換算電場E_r/N [Td]
    ez_face : ndarray of float64, shape (nr, nz+1)
        全軸方向面の法線方向換算電場E_z/N [Td]
    mobility_r : ndarray of float64, shape (nr+1, nz)
        半径面で評価した電子移動度 [m^2/(V s)]
    mobility_z : ndarray of float64, shape (nr, nz+1)
        軸方向面で評価した電子移動度 [m^2/(V s)]
    radial_boundary : ndarray of float64, shape (nz,)
        外周面の単位印加電圧に対する電位 [V/V]。長さnz
    upper_boundary : ndarray of float64, shape (nr,)
        上端面の単位印加電圧に対する電位 [V/V]。長さnr

    photoionization : PhotoionizationSolver
        共有格子で3項Helmholtz近似の光電離源を求めるソルバ。
    extra_species : dict
        反応機構の``species``節で追加した化学種の密度 {配列名: (nr, nz)} [m⁻³]。
        ``state.<配列名>``でも参照できる。
    ion_mobilities : tuple
        移動するイオンの (配列名, 電荷数, 移動度 [m²/(V s)]) の組。
    reaction_mask : ndarray of bool, shape (nr, nz)
        反応・光電離・背景電離を適用するセル。気体セルから上端の緩衝層を除いたもの。

    Notes
    -----
    通常はinitialize_simulation()で生成する。フィールドの再代入は禁止するが、配列要素はその場で更新する。
    中心配列は(nr,nz)、半径面は(nr+1,nz)、軸方向面は(nr,nz+1)。
    座標・セル幅・単位境界電位は対応方向の1次元、electron_lossesは長さ3。
    密度の単位はm⁻³、電位はV、換算電場はTd、速度はm/s。
    面積・体積は2πを省略し、electron_lossesだけは2πを含む実電子数。
    幾何配列はgridへの参照・ビューなので、格子を初期化後に書き換えない。
    状態退避は必要な数値配列をコピーする。LU分解を含む全体のdeepcopy/pickleは使用しない。
    ne・phiを含む全配列はスライス代入（state.ne[:] = ...）で更新する。

    Examples
    --------
    >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
    >>> config = SimulationConfig(nr=2, nz=3, voltage=0., nmax=0.,
    ...                           background_ionization=0., t_end=0.)
    >>> state = initialize_simulation(config)
    >>> state.ne.shape, state.er_face.shape, state.ez_face.shape
    ((2, 3), (3, 3), (2, 4))
    >>> state.volume is state.grid.volume
    True
    """
    poisson: "PoissonSolver"
    grid: CylindricalGrid  # Poisson・電場・輸送・刻み評価が共有する格子と面情報
    r: np.ndarray  # 半径方向の各セルの下側境界 [m]。最外側境界は含まない
    z: np.ndarray  # 軸方向の各セルの下側境界 [m]。最上側境界は含まない
    rh: np.ndarray  # 半径方向のセル中心座標 [m]
    zh: np.ndarray  # 軸方向のセル中心座標 [m]
    dr: np.ndarray  # 半径方向のセル幅 [m]
    dz: np.ndarray  # 軸方向のセル幅 [m]
    volume: np.ndarray  # セル体積係数 [m^3]（共通係数2πを省略）
    sr: np.ndarray  # 半径方向の下側セル面の面積係数 [m^2]（2πを省略）
    sz: np.ndarray  # 軸方向の下側セル面の面積係数 [m^2]（2πを省略）
    phi: np.ndarray  # 総電位 [V]。PoissonSolver.potentialと同じ配列
    ex: np.ndarray  # セル中心の半径方向換算電場E_r/N [Td]（名前のxはr方向）
    ey: np.ndarray  # セル中心の軸方向換算電場E_z/N [Td]（名前のyはz方向）
    abs_e: np.ndarray  # セル中心の換算電場の大きさ|E|/N [Td]。反応に使用
    n2: np.ndarray  # 中性窒素分子N₂の数密度 [m^-3]
    o2: np.ndarray  # 中性酸素分子O₂の数密度 [m^-3]
    n2p: np.ndarray  # 窒素分子正イオンN₂⁺の数密度 [m^-3]
    o2p: np.ndarray  # 酸素分子正イオンO₂⁺の数密度 [m^-3]
    om: np.ndarray  # 酸素原子負イオンO⁻の数密度 [m^-3]
    o2m: np.ndarray  # 酸素分子負イオンO₂⁻の数密度 [m^-3]
    ne: np.ndarray  # 電子密度 [m^-3]
    rho: np.ndarray  # Poisson電荷源 [V/m^2]。正味電荷密度/ε₀（半径重みなし）
    vex: np.ndarray  # 全半径面の電子ドリフト速度 [m/s]。形状(nr+1,nz)
    vey: np.ndarray  # 全軸方向面の電子ドリフト速度 [m/s]。形状(nr,nz+1)
    k1: np.ndarray  # N₂の電子衝突電離の反応係数 [m^3/s]
    k2: np.ndarray  # O₂への解離付着によってO⁻を生成する反応係数 [m^3/s]
    k3: np.ndarray  # O₂を第三体とする三体付着でO₂⁻を生成する反応係数 [m^6/s]
    gas_mask: np.ndarray  # 気体セル=True、電極セル=Falseのbool配列
    electron_losses: np.ndarray  # 累積流出電子数 [電極、接地平面、上端＋外周]（2πを含む）
    de: np.ndarray  # 気体セルの電子拡散係数 [m^2/s]。電極内は0。
    er_face: np.ndarray  # 全半径面の法線方向換算電場E_r/N [Td]
    ez_face: np.ndarray  # 全軸方向面の法線方向換算電場E_z/N [Td]
    mobility_r: np.ndarray  # 半径面で評価した電子移動度 [m^2/(V s)]
    mobility_z: np.ndarray  # 軸方向面で評価した電子移動度 [m^2/(V s)]
    radial_boundary: np.ndarray  # 外周面の単位印加電圧に対する電位 [V/V]。長さnz
    upper_boundary: np.ndarray  # 上端面の単位印加電圧に対する電位 [V/V]。長さnr
    o: np.ndarray
    k1_o2: np.ndarray | None
    chemistry: "ReactionTable | None"
    n2o: np.ndarray
    reaction_rates: np.ndarray | None
    photoionization: "PhotoionizationSolver"
    extra_species: dict  # 反応機構で追加した化学種の密度 {配列名: (nr, nz)} [m^-3]
    ion_mobilities: tuple  # 移動するイオン ((配列名, 電荷, 移動度 [m^2/(V s)]), ...)
    reaction_mask: np.ndarray  # 反応を適用するセル（気体かつ上端の緩衝層の外）

    def __getattr__(self, name):
        """追加化学種の密度を、基本種と同じく``state.<配列名>``で参照できるようにする。

        Examples
        --------
        >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
        >>> state = initialize_simulation(SimulationConfig(nr=2, nz=3, t_end=0.))
        >>> state.extra_species
        {}
        """
        extra = self.__dict__.get("extra_species")
        if extra is not None and name in extra:
            return extra[name]
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    @property
    def density_names(self):
        """時間発展する全化学種の配列名。基本9種の後に追加種が続く。

        Examples
        --------
        >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
        >>> initialize_simulation(SimulationConfig(nr=2, nz=3, t_end=0.)).density_names[:3]
        ('ne', 'n2', 'o2')
        """
        return DENSITY_NAMES + tuple(self.extra_species)

    @property
    def species_fields(self):
        """化学種名から配列名への対応（出力用）。

        Examples
        --------
        >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
        >>> initialize_simulation(SimulationConfig(nr=2, nz=3, t_end=0.)).species_fields['O2+']
        'o2p'
        """
        mechanism = None if self.chemistry is None else self.chemistry.mechanism
        if mechanism is not None and mechanism.tabulated:
            return dict(mechanism.species_fields)
        return dict(DRY_AIR_SPECIES_FIELDS)

    @property
    def charges(self):
        """荷電種の配列名から電荷数への対応。電荷密度の計算に使う。

        Examples
        --------
        >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
        >>> initialize_simulation(SimulationConfig(nr=2, nz=3, t_end=0.)).charges
        {'ne': -1, 'n2p': 1, 'o2p': 1, 'om': -1, 'o2m': -1}
        """
        mechanism = None if self.chemistry is None else self.chemistry.mechanism
        if mechanism is not None:
            return dict(mechanism.charges)
        return {"ne": -1, "n2p": 1, "o2p": 1, "om": -1, "o2m": -1}
