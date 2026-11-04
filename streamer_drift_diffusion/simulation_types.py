"""計算設定と状態。配列添字は [半径 i, 軸方向 j]、単位は SI（電場は Td）。"""
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from .grid import CylindricalGrid

if TYPE_CHECKING:
    from .electrostatics import PoissonSolver

# プロジェクトルートの絶対パス
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class SimulationConfig:
    """計算条件と既定値を集約する。単位はSI（電場はTd）。"""
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
    background_ionization: float = 1e13 # 毎更新の密度加算量。dt は掛けない。
    max_step: int = 5000 # 最大ステップ数
    write_every: int = 100 # データ書き出しの間隔（ステップ数）
    poisson_tolerance: float = 1e-10 # max|Aφ-b|/max|b| の許容値。旧SOR更新量とは異なる。
    adaptive_timestep: bool = False # Trueで適応刻み。固定モードは従来どおり。
    t_end: float | None = None # 適応モードの終了時刻 [s]。明示指定する。
    dt_max: float | None = None # 適応刻みの上限 [s]。Noneならdtを使用。
    dt_min: float = 1e-18 # 安定性のためにこれ未満が必要なら停止 [s]
    dt_safety_factor: float = 0.5 # 各安定条件へ掛ける安全係数
    dt_cfl_number: float = 0.8 # セルの流出率に対するCFL数
    dt_chemistry_fraction: float = 0.1 # 反応の生成・消費を制限する割合
    dt_growth_factor: float = 1.2 # 受理した刻みからの増加倍率の上限
    max_step_retries: int = 12 # 密度更新の縮小・再試行回数の上限
    adaptive_max_steps: int = 1000000 # 適応モードの受理step数の上限
    output_interval: float | None = None # 出力の実時間間隔 [s]。Noneならwrite_every*dt。
    electrode_gap: float = 1e-3 # 接地平面z=0から電極先端までの距離[m]。領域寸法とは独立。
    electron_diffusion: float = 0.0 # 検証用の一定電子拡散係数 [m^2/s]。0は拡散なし。


def molecular_density(config):
    """
    1 atm の分子数密度 [m^-3]。
    理想気体の状態方程式 PV = nRT より， n/V = P/(RT) = 1 atm / (気体定数×温度)
    アボガドロ数×1000 / (気体定数×温度)

    Args:
        config (SimulationConfig): SimulationConfig

    Returns:
        float: 1 atm の分子数密度 [m^-3]
    """
    return config.avogadro * 1000.0 / (config.gas_constant * config.temperature)


def gap_length(config):
    """
    ギャップ長の計算を行う

    Args:
        config (SimulationConfig): SimulationConfig

    Returns:
        float: 領域寸法とは独立した電極先端の高さ [m]。
    """
    return config.electrode_gap


# frozen=Trueにより，frozen dataclass となり，フィールドの再代入が禁止される
@dataclass(frozen=True)
class SimulationState:
    """フィールドの再代入を禁止し、配列要素をその場で更新する。

    電位・密度・ex/ey・abs_eはセル中心(nr,nz)。
    er_face・vex・mobility_rは半径面(nr+1,nz)、
    ez_face・vey・mobility_zは軸方向面(nr,nz+1)。
    座標・セル幅・外端の単位境界電位・electron_lossesは1次元。
    volume・sr・szは(nr,nz)。
    電極面はセル境界による階段状近似。gas_mask=Falseのセルを電極とする。
    2次元配列の添字[i, j]は半径方向i・軸方向jに対応する。
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
    phi: np.ndarray  # FiPyで直接求解した総電位 [V]
    ex: np.ndarray  # セル中心の半径方向換算電場E_r/N [Td]（名前のxはr方向）
    ey: np.ndarray  # セル中心の軸方向換算電場E_z/N [Td]（名前のyはz方向）
    abs_e: np.ndarray  # セル中心の換算電場の大きさ|E|/N [Td]。反応に使用
    n2: np.ndarray  # 中性窒素分子N₂の数密度 [m^-3]
    o2: np.ndarray  # 中性酸素分子O₂の数密度 [m^-3]
    n2p: np.ndarray  # 窒素分子正イオンN₂⁺の数密度 [m^-3]
    o2p: np.ndarray  # 酸素分子正イオンO₂⁺の数密度 [m^-3]
    om: np.ndarray  # 酸素原子負イオンO⁻の数密度 [m^-3]
    o2m: np.ndarray  # 酸素分子負イオンO₂⁻の数密度 [m^-3]
    ne: np.ndarray  # 電子数密度 [m^-3]
    rho: np.ndarray  # Poisson電荷源 [V/m^2]。正味電荷密度/ε₀（半径重みなし）
    vex: np.ndarray  # 全半径面の電子ドリフト速度 [m/s]。形状(nr+1,nz)
    vey: np.ndarray  # 全軸方向面の電子ドリフト速度 [m/s]。形状(nr,nz+1)
    k1: np.ndarray  # 電子衝突電離の反応係数 [m^3/s]（N₂・O₂に共通）
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
