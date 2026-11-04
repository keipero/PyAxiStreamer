"""FiPyによる総電位のPoisson求解、電荷源、換算電場。"""
import math
import numpy as np
from fipy import CellVariable, FaceVariable, DiffusionTerm, ImplicitSourceTerm
from fipy.solvers import LinearLUSolver

from .simulation_types import molecular_density
from .voltage_waveforms import applied_voltage


def boundary_values(state, voltage):
    """電極・接地・外端に指定する面電位。電場評価とPoissonで共有する。"""
    grid = state.grid
    result = np.full(grid.mesh.numberOfFaces, voltage, dtype=float)
    result[grid.face_ids[0][-1]] = voltage*state.radial_boundary
    result[grid.face_ids[1][:, 0]] = 0.
    result[grid.face_ids[1][:, -1]] = voltage*state.upper_boundary
    return result


class PoissonSolver:
    """共有メッシュ上で -∇²phi=rho を解く。境界は半セルDirichlet。

    気体間だけをDiffusionTermで結合し、吸収面と同じ距離で境界対角・源を作る。
    金属セルは独立した固定式へ切り離すため、大きな罰則係数は使わない。
    現在はFiPyのLU求解で、係数行列・分解の再利用を独自実装しない。
    """
    def __init__(self, grid, gas, config):
        self.grid, self.gas = grid, gas.copy()
        self.radial_boundary = np.array([hyperbolic_curve(1., config.cathode_radius,
            config.electrode_gap, config.nr*config.delta, z) for z in grid.zh])
        self.upper_boundary = np.array([hyperbolic_curve(1., config.cathode_radius,
            config.electrode_gap, r, config.nz*config.delta) for r in grid.rh])
        low, high = grid.adjacent(gas)
        self.boundary_rate = grid.area/grid.distances(gas)*(low ^ high)
        self.diagonal = np.where(gas, grid.sum_faces(self.boundary_rate)/grid.volume, 1/grid.spacing**2)
        self.potential = CellVariable(mesh=grid.mesh, value=0.)
        self.source = CellVariable(mesh=grid.mesh, value=0.)
        self.equation = (-DiffusionTerm(coeff=FaceVariable(mesh=grid.mesh, value=(low & high).astype(float)))
                         + ImplicitSourceTerm(coeff=CellVariable(mesh=grid.mesh, value=self.diagonal.ravel(order='F')))
                         == self.source)
        self.equation.cacheMatrix()
        self.equation.cacheRHSvector()
        self.tolerance = config.poisson_tolerance
        self.solver = LinearLUSolver(tolerance=self.tolerance, criterion='RHS')
        self.residual = 0.

    @property
    def boundary_source(self):
        grid = self.grid
        return np.where(self.gas,
            grid.sum_faces(self.boundary_rate*boundary_values(self, 1.))/grid.volume, self.diagonal)

    def solve(self, rho, voltage):
        """非有限入力・解・過大残差を拒否し、成功時だけ総電位を返す。"""
        if not np.isfinite(voltage) or not np.all(np.isfinite(rho)):
            raise RuntimeError("nonfinite Poisson source or voltage")
        self.source.setValue((np.where(self.gas, rho, 0.) + voltage*self.boundary_source).ravel(order='F'))
        # LUには初期推定が不要。前stepが近い場合も微小な電荷変化を省略させない。
        self.potential.setValue(0.)
        self.equation.solve(var=self.potential, solver=self.solver)
        values = np.asarray(self.potential.value)
        rhs = self.equation.RHSvector
        residual = self.equation.matrix*values-rhs
        self.residual = float(np.max(np.abs(residual))/max(float(np.max(np.abs(rhs))), np.finfo(float).tiny))
        if not np.all(np.isfinite(values)) or not np.isfinite(self.residual) or self.residual > self.tolerance:
            raise RuntimeError(f"Poisson solve failed residual check: {self.residual:.6e}")
        result = values.reshape(self.grid.shape, order='F').copy()
        result[~self.gas] = voltage
        return result


def calculate_potential(state, config, *, verbose=True, time=0.0, waveform=None):
    """現在の電荷と境界電圧から総電位を求め、印加電圧を返す。"""
    voltage = applied_voltage(config, time, waveform=waveform)
    state.phi[:] = state.poisson.solve(state.rho, voltage)
    return voltage


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
    boundary = boundary_values(state, voltage)
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
