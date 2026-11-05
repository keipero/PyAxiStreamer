"""FiPyによる総電位のPoisson求解、電荷源、換算電場。"""
import math
import numpy as np
from fipy import CellVariable, FaceVariable, DiffusionTerm, ImplicitSourceTerm
from fipy.solvers import LinearLUSolver, solver_suite

from .simulation_types import molecular_density
from .voltage_waveforms import applied_voltage


def boundary_values(state, voltage):
    """電極・接地・外端に指定する面電位を作る。

    Parameters
    ----------
    state : SimulationState or PoissonSolver
        grid、radial_boundary、upper_boundaryを保持するオブジェクト。
    voltage : float
        現在の印加電圧 [V]。

    Returns
    -------
    ndarray, shape (nfaces,)
        共有メッシュ面番号順の境界電位 [V]。

    Notes
    -----
    電極はvoltage、接地は0 V、外周・上端は単位電圧応答×voltage。
    内部の気体－気体面や対称軸にも補助値を含むため、使用時は面分類・面積で有効面を選ぶ。

    Examples
    --------
    >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
    >>> config = SimulationConfig(nr=2, nz=3, voltage=0., nmax=0.,
    ...                           background_ionization=0., t_end=0.)
    >>> state = initialize_simulation(config)
    >>> values = boundary_values(state, 0.)
    >>> float(values.max())
    0.0
    """
    grid = state.grid
    result = np.full(grid.mesh.numberOfFaces, voltage, dtype=float)
    result[grid.face_ids[0][-1]] = voltage*state.radial_boundary
    result[grid.face_ids[1][:, 0]] = 0.
    result[grid.face_ids[1][:, -1]] = voltage*state.upper_boundary
    return result


class PoissonSolver:
    """共有円筒メッシュ上で総電位を直接解くFiPy Poissonソルバー。

    Parameters
    ----------
    grid : CylindricalGrid
        Poisson・電場・輸送が共有する円筒格子。
    gas : ndarray of bool, shape (nr, nz)
        気体True、電極False。初期化後の形状は固定する。
    config : SimulationConfig
        物理定数、電圧、時間刻み・安定条件を保持する設定。
    reuse_matrix : bool, optional
        PETScの行列・LU分解を再利用する。既定True。SciPyではFalseとして扱う。

    Attributes
    ----------
    potential : fipy.CellVariable
        FiPy面積・体積積分系で解くセル中心電位 [V]。
    source : fipy.CellVariable
        電荷源と境界源を合成した右辺のセル係数 [V/m²]。
    radial_boundary, upper_boundary : ndarray
        外周・上端の単位印加電圧に対する電位応答 [V/V]。
    boundary_rate : ndarray, shape (nfaces,)
        気体に接する境界面のA/d [m]。2πを省略する。
    diagonal : ndarray, shape (nr, nz)
        境界対角係数 [m⁻²]。金属セルは独立固定式の係数。
    equation : fipy.terms.term.Term
        FiPyで組み立てる総電位の方程式。
    solver : fipy.solvers.LinearLUSolver or CachedLUSolver
        選択されたFiPyバックエンドのLUソルバー。PETScの既定は固定行列・LUを再利用する。
    reuse_matrix : bool
        実際に固定行列・LUを再利用するかどうか。初期化後に変更しない。
    tolerance, residual : float
        許容値と直近のmax(abs(A*phi-b))/max(abs(b))。

    Notes
    -----
    方程式は -laplacian(phi)=rho。rhoは電荷密度/epsilon0 [V/m²]。
    気体間だけをDiffusionTermで結合し、境界は共有の面積・半セル距離から対角項と源を作る。
    金属セルは気体から切り離した独立固定式で、大きな罰則係数を使わない。
    PETScの既定ではFiPyで一度だけ行列を組み、右辺を更新して同じKSP・LU分解を使う。
    SciPyまたはreuse_matrix=FalseではFiPyの通常の組立・求解を毎回行う。
    設定・気体マスクの検査はinitialize_simulation()が担当する。内部電極形状は求解中に変えない。
    格子・係数・電極形状の変更には新しいPoissonSolverを作る。外端電位応答の変更は右辺に反映する。
    close()でPETScリソースを解放でき、通常の時間ループは成功・失敗とも終了時に呼ぶ。

    Examples
    --------
    >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
    >>> config = SimulationConfig(nr=2, nz=3, voltage=0., nmax=0.,
    ...                           background_ionization=0., t_end=0.)
    >>> state = initialize_simulation(config)
    >>> solver = PoissonSolver(state.grid, state.gas_mask, config)
    >>> import numpy as np
    >>> answer = solver.solve(np.zeros(state.ne.shape), 0.)
    >>> float(answer.max())
    0.0
    """
    def __init__(self, grid, gas, config, *, reuse_matrix=True):
        """共有メッシュ上に境界対角・境界源とFiPy方程式を準備する。

        Parameters
        ----------
        grid : CylindricalGrid
            Poisson・電場・輸送が共有する円筒格子。
        gas : ndarray of bool, shape (nr, nz)
            気体True、電極False。初期化後の形状は固定する。
        config : SimulationConfig
            物理定数、電圧、時間刻み・安定条件を保持する設定。
        reuse_matrix : bool, optional
            PETScで固定行列・LUを再利用するかどうか。既定True。比較計測時はFalseを指定できる。

        Notes
        -----
        この時点では電位を求解しない。外端電位は双曲線電極の解析式から単位応答を作る。

        Examples
        --------
        >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
        >>> config = SimulationConfig(nr=2, nz=3, voltage=0., nmax=0.,
        ...                           background_ionization=0., t_end=0.)
        >>> state = initialize_simulation(config)
        >>> solver = PoissonSolver(state.grid, state.gas_mask, config)
        >>> solver.potential.mesh is state.grid.mesh
        True
        """
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
        self.reuse_matrix = reuse_matrix and solver_suite == 'petsc'
        if self.reuse_matrix:
            from ._petsc import CachedLUSolver
        self.solver = (CachedLUSolver if self.reuse_matrix else LinearLUSolver)(tolerance=self.tolerance, criterion='RHS')
        self.residual = 0.

    @property
    def boundary_source(self):
        """単位印加電圧に対するセルごとの境界源を返す。

        Returns
        -------
        ndarray, shape (nr, nz)
            印加電圧を掛ける前の境界源 [m⁻²]。金属は独立固定式の対角係数。

        Notes
        -----
        radial_boundary・upper_boundaryの現在値を参照して計算する。

        Examples
        --------
        >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
        >>> config = SimulationConfig(nr=2, nz=3, voltage=0., nmax=0.,
        ...                           background_ionization=0., t_end=0.)
        >>> state = initialize_simulation(config)
        >>> state.poisson.boundary_source.shape
        (2, 3)
        """
        grid = self.grid
        return np.where(self.gas,
            grid.sum_faces(self.boundary_rate*boundary_values(self, 1.))/grid.volume, self.diagonal)

    def solve(self, rho, voltage):
        """有限性と残差を検査し、指定した電荷源・電圧の総電位を返す。

        Parameters
        ----------
        rho : ndarray, shape (nr, nz)
            電荷密度/epsilon0 [V/m²]。有限値が必要で、金属セルの値は源に含めない。
        voltage : float
            電極の印加電圧 [V]。有限値を指定する。

        Returns
        -------
        ndarray, shape (nr, nz)
            potentialの正本とメモリを共有する解の2次元ビュー [V]。
            金属セルは指定電圧に揃える。求解をまたいで退避する値にはcopy()を使う。

        Raises
        ------
        RuntimeError
            非有限入力、非有限解、または相対残差が許容値を超える場合。

        Notes
        -----
        LUには初期推定が不要なため初期値を0にする。境界右辺に比べて小さい電荷変化も省略しない。
        PETSc再利用時はFiPyの保存済み行列を保持し、体積を掛けた右辺だけを更新する。
        FiPy 4.0.3の_storeMatrix()と_solve()へ渡し、通常と同じ正本・ベクトル変換を使う。
        残差の右辺が0の場合、分母下限にはnp.finfo(float).tinyを使う。
        potentialはstate.phi_varと同じ正本。残差異常・求解例外では求解前の電位へ復元しKSPを解放する。
        退避コピーはこの呼出し中だけ保持し、解を別の状態配列へコピーしない。

        Examples
        --------
        >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
        >>> config = SimulationConfig(nr=2, nz=3, voltage=0., nmax=0.,
        ...                           background_ionization=0., t_end=0.)
        >>> state = initialize_simulation(config)
        >>> import numpy as np
        >>> answer = state.poisson.solve(np.zeros(state.ne.shape), 0.)
        >>> float(answer.max())
        0.0
        >>> state.poisson.residual <= config.poisson_tolerance
        True
        """
        if not np.isfinite(voltage) or not np.all(np.isfinite(rho)):
            raise RuntimeError("nonfinite Poisson source or voltage")
        self.source.setValue((np.where(self.gas, rho, 0.) + voltage*self.boundary_source).ravel(order='F'))
        previous = self.potential.value.copy()
        try:
            # LUはゼロ初期値から解き、微小な電荷変化も省略しない。
            self.potential.setValue(0.)
            if self.reuse_matrix and self.equation.matrix is not None:
                self.equation.RHSvector[:] = self.source.value*self.grid.mesh.cellVolumes
                self.solver._storeMatrix(self.potential, self.equation.matrix, self.equation.RHSvector)
                self.solver._solve()
            else:
                self.equation.solve(var=self.potential, solver=self.solver)
            values = self.potential.value
            rhs = self.equation.RHSvector
            residual = self.equation.matrix*values-rhs
            self.residual = float(np.max(np.abs(residual))/max(float(np.max(np.abs(rhs))), np.finfo(float).tiny))
            if not np.all(np.isfinite(values)) or not np.isfinite(self.residual) or self.residual > self.tolerance:
                raise RuntimeError(f"Poisson solve failed residual check: {self.residual:.6e}")
            self.potential.setValue(voltage, where=~self.gas.ravel(order='F'))
        except Exception:
            self.potential.setValue(previous)
            self.close()
            raise
        return self.potential.value.reshape(self.grid.shape, order='F')

    def close(self):
        """保持するPETScのLU分解と一時ベクトルを解放する。

        Returns
        -------
        None
            SciPyでは何もしない。電位・FiPy行列は保持し、再求解時にKSPを再作成できる。

        Notes
        -----
        何度呼んでもよい。固定係数を変更するためのAPIではない。
        単独使用では明示的に呼ぶ。通常のrun_simulation()は終了時に自動で呼ぶ。

        Examples
        --------
        >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
        >>> state = initialize_simulation(SimulationConfig(nr=2, nz=3, t_end=0.))
        >>> state.poisson.close()
        >>> state.poisson.close()
        """
        if self.reuse_matrix:
            self.solver.close()


def calculate_potential(state, config, *, verbose=True, time=0.0, waveform=None):
    """現在の電荷源と電圧波形から総電位を計算する。

    Parameters
    ----------
    state : SimulationState
        共有格子と計算配列。中心配列は(nr,nz)、面配列は方向別の全面形状。
    config : SimulationConfig
        物理定数、電圧、時間刻み・安定条件を保持する設定。
    verbose : bool, optional
        呼出側の表示設定。現在のLU求解は反復進捗を表示しない。
    time : float, optional
        電圧を評価する時刻 [s]。既定0。
    waveform : VoltageWaveform, optional
        読み込み済みのインパルス波形。Noneなら必要時に設定ファイルを読む。

    Returns
    -------
    float
        評価した印加電圧 [V]。state.phiを同じ時刻の解へ更新する。

    Raises
    ------
    RuntimeError
        Poisson求解の有限性・残差検査が失敗。
    ValueError
        波形・電圧モード・評価時刻が不正。
    OSError
        波形ファイルの読込みに失敗。

    Notes
    -----
    密度更新後に呼ぶ場合、calculate_charge_density()でstate.rhoを先に更新する。
    旧lphi/pphiの合成は行わない。

    Examples
    --------
    >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
    >>> config = SimulationConfig(nr=2, nz=3, voltage=0., nmax=0.,
    ...                           background_ionization=0., t_end=0.)
    >>> state = initialize_simulation(config)
    >>> calculate_charge_density(state, config)
    >>> calculate_potential(state, config, time=0., verbose=False)
    0.0
    >>> float(state.phi.max())
    0.0
    """
    voltage = applied_voltage(config, time, waveform=waveform)
    state.poisson.solve(state.rho, voltage)
    return voltage


def hyperbolic_curve(vele, cr, b, r, z):
    """双曲線電極と接地平面による基礎電位を1点で評価する。

    Parameters
    ----------
    vele : float
        双曲線電極の印加電圧 [V]。
    cr : float
        電極先端の曲率半径 [m]。正の値を指定する。
    b : float
        接地平面から電極先端までのギャップ長 [m]。
        正の値を指定する。
    r : float
        評価点の半径方向座標 [m]。
    z : float
        評価点の軸方向座標 [m]。接地平面をz=0とする。

    Returns
    -------
    float
        評価点の基礎電位 [V]。

    Notes
    -----
    a = sqrt(cr * b)とし、(z/b)^2 - (r/a)^2 >= 1の電極領域では
    veleを返す。それ以外のz=0では0を返し、気体領域では
    双曲線電極の解析式を使う。空間電荷の寄与は含まない。

    Examples
    --------
    >>> hyperbolic_curve(4000., 300e-6, 1e-3, 0., 0.)
    0.0
    >>> hyperbolic_curve(4000., 300e-6, 1e-3, 0., 1e-3)
    4000.0
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

    Parameters
    ----------
    state : SimulationState
        ne、n2p、o2p、om、o2m [m^-3]と気体マスク。
    config : SimulationConfig
        電気素量と真空の誘電率。

    Returns
    -------
    None
        state.rhoをその場で更新する。電極内は0。

    Notes
    -----
    rho = e/ε₀ * (n2p+o2p-om-o2m-ne)。半径重みは掛けない。
    円筒座標の幾何は有限体積Poisson係数側へ含める。

    Examples
    --------
    >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
    >>> config = SimulationConfig(nr=2, nz=3, voltage=0., nmax=0.,
    ...                           background_ionization=0., t_end=0.)
    >>> state = initialize_simulation(config)
    >>> calculate_charge_density(state, config)
    >>> float(state.rho.max())
    0.0
    """
    positive = state.n2p + state.o2p
    negative = state.om + state.o2m + state.ne
    state.rho[:] = (config.elementary_charge / config.vacuum_permittivity) * (positive - negative)
    state.rho[~state.gas_mask] = 0.


def calculate_electric_field(state, config, *, voltage=None):
    """全セル面の法線電場と、反応・出力用のセル中心電場を求める。

    Parameters
    ----------
    state : SimulationState
        中心電位phi [V]、気体マスク、単位境界電位。
    config : SimulationConfig
        換算に用いるガス条件。
    voltage : float, optional
        現在の印加電圧 [V]。省略時はconfig.voltage。
        インパルス計算ではcalculate_potential()が返す実際の電圧を渡す。

    Returns
    -------
    None
        er_face (nr+1,nz)、ez_face (nr,nz+1)と、
        中心ex・ey・abs_e (nr,nz) [Td]を更新する。

    Notes
    -----
    面法線電場はPoissonの有限体積係数と同じ距離・境界電位を使う。
    軸は法線電場0。電極・接地・上端・外周は半セル幅のDirichlet面。
    中心成分は対向する2面の算術平均で、中心abs_eはその2成分から求める。
    電極内部の面電場・中心電場は0、気体・電極間の法線電場は保持する。
    等間隔格子を対象とする。電場の単位は全てTd。

    Examples
    --------
    >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
    >>> config = SimulationConfig(nr=2, nz=3, voltage=0., nmax=0.,
    ...                           background_ionization=0., t_end=0.)
    >>> state = initialize_simulation(config)
    >>> calculate_charge_density(state, config)
    >>> voltage = calculate_potential(state, config, verbose=False)
    >>> calculate_electric_field(state, config, voltage=voltage)
    >>> state.er_face.shape, state.ez_face.shape
    ((3, 3), (2, 4))
    """
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
