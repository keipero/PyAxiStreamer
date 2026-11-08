"""計算設定と状態。配列添字は [半径 i, 軸方向 j]、単位は SI（電場は Td）。"""
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from fipy import CellVariable
from .grid import CylindricalGrid

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
        半径方向の分割数 Default: ``200``.
    nz : int, optional
        軸方向の分割数。既定の上端は1.5 mm、先端は1 mm。 Default: ``150``.
    delta : float, optional
        格子間隔 [m] Default: ``1e-05``.
    temperature : float, optional
        温度 [K] Default: ``300.0``.
    avogadro : float, optional
        アボガドロ数 [mol^-1] Default: ``6.022141e+23``.
    gas_constant : float, optional
        気体定数 [L·atm/(K·mol)] Default: ``0.08205736608096``.
    elementary_charge : float, optional
        電子の電荷 [C] Default: ``1.602176462e-19``.
    vacuum_permittivity : float, optional
        真空の誘電率 [F/m] Default: ``8.85e-12``.
    oxygen_fraction : float, optional
        酸素分率 Default: ``0.2``.
    nitrogen_fraction : float, optional
        窒素分率 Default: ``0.8``.
    cathode_radius : float, optional
        カソード半径 [m] Default: ``0.0003``.
    voltage : float, optional
        DCモードの電圧 [V] Default: ``4000.0``.
    voltage_mode : str, optional
        電圧モード（"dc" または "impulse"） Default: ``'dc'``.
    impulse_peak_voltage : float, optional
        インパルス電圧のピーク値 [V] Default: ``5000.0``.
    impulse_time_scale : float, optional
        インパルス電圧の時間倍率（無次元） Default: ``1.0``.
    voltage_waveform_file : str, optional
        インパルス電圧波形ファイルのパス
    dt : float, optional
        時間刻み [s] Default: ``1e-12``.
    nmax : float, optional
        最大電子密度 [m^-3] Default: ``1e+19``.
    z0 : float, optional
        初期電子密度の中心位置 [m] Default: ``0.0009``.
    sr0 : float, optional
        初期電子密度の半径方向の広がり [m] Default: ``0.0001``.
    sz0 : float, optional
        初期電子密度の軸方向の広がり [m] Default: ``0.0001``.
    background_ionization : float, optional
        基準幅dtに対応する背景の密度加算量 [m⁻³]。実際はactual_dt/dtを掛けて供給する。 Default: ``10000000000000.0``.
    poisson_tolerance : float, optional
        max|Aφ-b|/max|b| の許容値。旧SOR更新量とは異なる。 Default: ``1e-10``.
    adaptive_timestep : bool, optional
        共通ループで刻み幅の選択・再試行だけを切り替える。 Default: ``False``.
    t_end : float, optional
        両モード共通の終了時刻 [s] Default: ``5.001e-09``.
    dt_max : float or None, optional
        適応刻みの上限 [s]。Noneならdt。固定刻みの基本幅には使用しない。 Default: ``None``.
    dt_min : float, optional
        適応候補刻みの停止下限 [s]。イベントへの安全な端数短縮は下限未満も許す。 Default: ``1e-18``.
    dt_safety_factor : float, optional
        各安定条件へ掛ける安全係数 Default: ``0.5``.
    dt_cfl_number : float, optional
        セルの流出率に対するCFL数 Default: ``0.8``.
    dt_chemistry_fraction : float, optional
        反応の生成・消費を制限する割合 Default: ``0.1``.
    dt_growth_factor : float, optional
        受理した刻みからの増加倍率の上限 Default: ``1.2``.
    max_step_retries : int, optional
        密度更新の縮小・再試行回数の上限 Default: ``12``.
    max_steps : int, optional
        両モード共通の受理step数の停止上限 Default: ``1000000``.
    output_interval : float, optional
        出力の実時間間隔 [s] Default: ``1e-10``.
    electrode_gap : float, optional
        接地平面z=0から電極先端までの距離[m]。領域寸法とは独立。 Default: ``0.001``.
    electron_diffusion : float, optional
        従来モードの一定電子拡散係数 [m²/s]。0は拡散なし。
        BOLOS輸送列のある表を使う場合は既定値0を指定する。
    chemistry_table : str or pathlib.Path or None, optional
        Py3BOLOSとCanteraで生成した電子輸送・反応係数表。Noneなら従来の経験式。

    photoionization_efficiency : float, optional
        発光源の無次元係数。消光補正前の値で、範囲は(0, 1]、既定0.075。

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
    voltage_mode: str = "dc" # 電圧モード（"dc" または "impulse"）
    impulse_peak_voltage: float = 5000.0 # インパルス電圧のピーク値 [V]
    impulse_time_scale: float = 1.0 # インパルス電圧の時間倍率（無次元）
    voltage_waveform_file: str = str(Path(__file__).resolve().parent / "data" / "impulse_waveform_V50_tr5_tw6.txt") # インパルス電圧波形ファイルのパス
    dt: float = 1.0e-12 # 時間刻み [s]
    nmax: float = 1e19 # 最大電子密度 [m^-3]
    z0: float = 900e-6 # 初期電子密度の中心位置 [m]
    sr0: float = 100e-6 # 初期電子密度の半径方向の広がり [m]
    sz0: float = 100e-6 # 初期電子密度の軸方向の広がり [m]
    background_ionization: float = 1e13 # 基準幅dtの密度加算量。実際はactual_dt/dtを掛ける。
    poisson_tolerance: float = 1e-10 # max|Aφ-b|/max|b| の許容値。旧SOR更新量とは異なる。
    adaptive_timestep: bool = False # 共通ループで刻み幅の選択・再試行だけを切り替える。
    t_end: float = 5.001e-9 # 両モード共通の終了時刻 [s]
    dt_max: float | None = None # 適応刻みの上限 [s]。Noneならdtを使用。
    dt_min: float = 1e-18 # 安定性のためにこれ未満が必要なら停止 [s]
    dt_safety_factor: float = 0.5 # 各安定条件へ掛ける安全係数
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
    """共有格子・FiPyソルバーと、時間発展する物理量配列を保持する。

    Parameters
    ----------
    poisson : PoissonSolver
        総電位を直接求解するFiPy PoissonSolver。
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
    phi_var : fipy.CellVariable
        総電位の正本 [V]。poisson.potentialと共有するFiPyセル番号順の変数。
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
    ne_var : fipy.CellVariable
        電子密度の正本 [m⁻³]。輸送と共有するFiPyセル番号順の変数。hasOld=True。
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

    Attributes
    ----------
    ne, phi : ndarray of float64, shape (nr, nz)
        正本の値をFortran順に変形した2次元ビュー。独立した値は保持しない。
        neは電子密度 [m⁻³]、phiは総電位 [V]。独立した退避値にはcopy()を使う。

    photoionization : PhotoionizationSolver
        共有格子で3項Helmholtz近似の光電離源を求めるソルバ。

    Notes
    -----
    通常はinitialize_simulation()で生成する。フィールドの再代入は禁止するが、配列要素はその場で更新する。
    中心配列は(nr,nz)、半径面は(nr+1,nz)、軸方向面は(nr,nz+1)。
    座標・セル幅・単位境界電位は対応方向の1次元、electron_lossesは長さ3。
    密度の単位はm⁻³、電位はV、換算電場はTd、速度はm/s。
    面積・体積は2πを省略し、electron_lossesだけは2πを含む実電子数。
    幾何配列はgridへの参照・ビューなので、格子を初期化後に書き換えない。
    状態退避は必要な数値配列をコピーする。FiPyの方程式を含む全体のdeepcopy/pickleは使用しない。
    ne・phiへのNumPyスライス代入は正本の値を変える。FiPyの派生式のキャッシュも
    更新する場合はne_var・phi_varのsetValue()を使う。組込みの初期化・反応・復元はこのAPIを使う。

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
    phi_var: CellVariable  # 総電位の正本。PoissonSolverと共有する。
    ex: np.ndarray  # セル中心の半径方向換算電場E_r/N [Td]（名前のxはr方向）
    ey: np.ndarray  # セル中心の軸方向換算電場E_z/N [Td]（名前のyはz方向）
    abs_e: np.ndarray  # セル中心の換算電場の大きさ|E|/N [Td]。反応に使用
    n2: np.ndarray  # 中性窒素分子N₂の数密度 [m^-3]
    o2: np.ndarray  # 中性酸素分子O₂の数密度 [m^-3]
    n2p: np.ndarray  # 窒素分子正イオンN₂⁺の数密度 [m^-3]
    o2p: np.ndarray  # 酸素分子正イオンO₂⁺の数密度 [m^-3]
    om: np.ndarray  # 酸素原子負イオンO⁻の数密度 [m^-3]
    o2m: np.ndarray  # 酸素分子負イオンO₂⁻の数密度 [m^-3]
    ne_var: CellVariable  # 電子密度の正本。輸送と共有し、旧値も保持する。
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
    photoionization: "PhotoionizationSolver"

    @property
    def ne(self):
        """電子密度の正本を、コピーなしのセル配列として参照する。

        Returns
        -------
        ndarray, shape (nr, nz)
            ne_var.valueとメモリを共有する電子密度 [m⁻³]。独立した退避にはcopy()を使う。

        Notes
        -----
        FiPyセル番号との対応はFortran順。NumPyでの要素変更は正本に反映する。
        FiPyの派生式にも変更を通知する代入にはne_var.setValue()を用いる。

        Examples
        --------
        >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
        >>> import numpy as np
        >>> state = initialize_simulation(SimulationConfig(nr=2, nz=3, t_end=0.))
        >>> np.shares_memory(state.ne, state.ne_var.value)
        True
        """
        return self.ne_var.value.reshape(self.grid.shape, order='F')

    @property
    def phi(self):
        """総電位の正本を、コピーなしのセル配列として参照する。

        Returns
        -------
        ndarray, shape (nr, nz)
            phi_var.valueとメモリを共有する総電位 [V]。求解後も最新値を参照する。

        Notes
        -----
        Fortran順の2次元ビュー。独立した退避にはcopy()、FiPyの派生式への変更通知を
        伴う代入にはphi_var.setValue()を使う。phi_varはpoisson.potentialと同一の変数。

        Examples
        --------
        >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
        >>> import numpy as np
        >>> state = initialize_simulation(SimulationConfig(nr=2, nz=3, t_end=0.))
        >>> state.phi_var is state.poisson.potential
        True
        >>> np.shares_memory(state.phi, state.phi_var.value)
        True
        """
        return self.phi_var.value.reshape(self.grid.shape, order='F')
