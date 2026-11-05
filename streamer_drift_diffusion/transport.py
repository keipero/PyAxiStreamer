"""FiPyによる円筒有限体積の陽的電子移流拡散と、吸収境界の粒子収支。"""
import math
import numpy as np

from fipy import (CellVariable, ExplicitDiffusionTerm, ExplicitUpwindConvectionTerm,
                  FaceVariable, TransientTerm)
from fipy.solvers import LinearLUSolver, solver_suite



class _ExplicitUpwind(ExplicitUpwindConvectionTerm):
    """FiPy 4.0.xでゼロ面の不要なPeclet数除算を避ける上流項。

    Parameters
    ----------
    coeff : fipy.FaceVariable
        気体間面の速度ベクトル [m/s]。

    Notes
    -----
    正のTransientTermに対する一次上流の重みは速度の符号だけで決まる。
    拡散幾何係数を重み計算へ渡さず、D=0・速度0の面や軸で0/0を起こさない。
    物理係数やNumPyの例外設定は変更しない。内部実装用で、通常はElectronTransportを使う。

    Examples
    --------
    >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
    >>> config = SimulationConfig(nr=2, nz=3, voltage=0., nmax=0.,
    ...                           background_ionization=0., t_end=0.)
    >>> state = initialize_simulation(config)
    >>> transport = ElectronTransport(state, config.delta)
    >>> import numpy as np
    >>> with np.errstate(over='raise', invalid='raise', divide='raise'):
    ...     losses = transport.step(state, config.dt)
    >>> float(losses.sum())
    0.0
    """
    def _getWeight(self, var, transientGeomCoeff=None, diffusionGeomCoeff=None):
        """拡散係数に依存しない一次上流の重みをFiPyへ要求する。

        Parameters
        ----------
        var : fipy.CellVariable
            輸送されるセル中心変数。
        transientGeomCoeff : array_like, optional
            FiPyが渡す時間項の幾何係数。
        diffusionGeomCoeff : array_like, optional
            FiPy互換の引数。この実装では常に無視してNoneを渡す。

        Returns
        -------
        dict
            FiPy内部の面重み。具体的な構造はFiPyの実装に従う。

        Notes
        -----
        FiPy 4.0.x内部APIへの限定したアダプタ。単独で呼ばず、ElectronTransportから使用する。
        """
        return super()._getWeight(var, transientGeomCoeff=transientGeomCoeff,
                                  diffusionGeomCoeff=None)


class ElectronTransport:
    """共有格子上のFiPy電子移流拡散と、吸収面の流出数を計算する。

    Parameters
    ----------
    state : SimulationState
        共有格子と計算配列。中心配列は(nr,nz)、面配列は方向別の全面形状。
    spacing : float
        セル幅 [m]。共有格子のspacingに一致させる。

    Raises
    ------
    ValueError
        spacingと共有格子のセル幅が一致しない。

    Attributes
    ----------
    grid, mesh : CylindricalGrid and fipy.meshes.mesh.Mesh
        state.gridとそのメッシュへの参照。新たなメッシュは生成しない。
    density : fipy.CellVariable
        state.ne_varと同一の電子密度の正本 [m⁻³]。各試行の開始時の旧値を保持する。
    velocity : fipy.FaceVariable
        気体間面の速度ベクトル [m/s]。
    diffusion : fipy.FaceVariable
        気体間面の拡散係数 [m²/s]。
    absorption : fipy.CellVariable
        セルの吸収損失係数 [s⁻¹]。
    gas : ndarray of bool, shape (nr, nz)
        初期の気体マスクのコピー。形状変更を検出する。
    interior, boundary, low_gas : ndarray of bool, shape (nfaces,)
        初期化時に固定する気体間面、吸収境界面、小さい座標側の気体区分。
    distance : ndarray, shape (nfaces,)
        気体間はセル幅、その他は半セル幅の距離 [m]。
    diffusion_rates, boundary_rates : ndarray, shape (nfaces,)
        prepare()で生成するD*A/dと、外向き移流を加えた吸収係数 [m³/s]。
        共通係数2πは省略する。boundary_ratesは境界面以外で0。
    transport_limit, diffusion_limit : float
        prepare()後の合算輸送・純拡散の陽的上限 [s]。損失0ならmath.inf。
        CFL数・共通安全係数を掛ける前の値。

    Notes
    -----
    中心配列は(nr,nz)、vexは(nr+1,nz)、veyは(nr,nz+1)。deは等方拡散係数 [m²/s]。
    一次上流移流と拡散を同じ旧密度から陽的Eulerで更新する。
    気体間Dは調和平均。電極・外端は気体側Dと半セル距離による吸収で、外から供給しない。
    1シミュレーションにつき一度作り、格子と電極形状は計算中に変更しない。
    流出数の累積は受理を判断する呼出側が担当する。
    時間ループはprepare()の面係数をCFL・輸送・流出数へ共用する。
    単独のstep()は既定で係数を再評価し、速度・Dの変更を反映する。
    未知量の行列は正の対角行列。PETScではpreonly/Jacobiによる対角求解を使い、
    微小な係数がLUの絶対ゼロピボット閾値に掛かることを避ける。SciPyではLUを使う。
    いずれも初期残差を基準に解き、右辺に比べて小さい輸送変化も省略しない。
    FiPyの陽的拡散項は組立時の現在値を使うため、求解前に現在値をゼロにしない。
    全輸送係数が0なら現在値をそのまま保持し、不要な対角求解による丸め電荷を作らない。

    Examples
    --------
    >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
    >>> config = SimulationConfig(nr=2, nz=3, voltage=0., nmax=0.,
    ...                           background_ionization=0., t_end=0.)
    >>> state = initialize_simulation(config)
    >>> transport = ElectronTransport(state, config.delta)
    >>> losses = transport.step(state, config.dt)
    >>> losses.shape
    (3,)
    >>> float(losses.sum())
    0.0
    """
    def __init__(self, state, spacing):
        """共有メッシュを再利用し、輸送変数と陽的方程式を生成する。

        Parameters
        ----------
        state : SimulationState
            共有格子と計算配列。中心配列は(nr,nz)、面配列は方向別の全面形状。
        spacing : float
            セル幅 [m]。共有格子のspacingに一致させる。

        Raises
        ------
        ValueError
            spacingとstate.grid.spacingが一致しない。

        Notes
        -----
        ここでは密度更新・流出加算は行わない。

        Examples
        --------
        >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
        >>> config = SimulationConfig(nr=2, nz=3, voltage=0., nmax=0.,
        ...                           background_ionization=0., t_end=0.)
        >>> state = initialize_simulation(config)
        >>> transport = ElectronTransport(state, config.delta)
        >>> transport.mesh is state.grid.mesh
        True
        """
        self.grid = state.grid
        if spacing != self.grid.spacing:
            raise ValueError("transport spacing must match the shared grid")
        self.shape, self.mesh = self.grid.shape, self.grid.mesh
        self.gas = state.gas_mask.copy()
        self.low_gas, high_gas = self.grid.adjacent(self.gas)
        self.interior, self.boundary = self.low_gas & high_gas, self.low_gas ^ high_gas
        self.distance = self.grid.distances(self.gas)
        self.density = state.ne_var
        self.velocity = FaceVariable(mesh=self.mesh, rank=1, value=0.)
        self.diffusion = FaceVariable(mesh=self.mesh, value=0.)
        self.absorption = CellVariable(mesh=self.mesh, value=0.)
        self.equation = (TransientTerm()
                         == ExplicitDiffusionTerm(coeff=self.diffusion)
                         - _ExplicitUpwind(coeff=self.velocity)
                         - self.absorption * self.density.old)
        self.solver = LinearLUSolver(tolerance=1e-12, criterion='initial')
        if solver_suite == 'petsc':
            from fipy.solvers.petsc import JacobiPreconditioner
            # FiPy 4.0.3のLUクラスはprecon引数を固定するため、生成後に対角PCを設定する。
            self.solver.preconditioner = JacobiPreconditioner()

    def prepare(self, state):
        """現在の速度・Dから、CFL・輸送・境界流出が共有する面係数を準備する。

        Parameters
        ----------
        state : SimulationState
            共有格子、固定のgas_mask、全半径面vex (nr+1,nz)、全軸方向面vey
            (nr,nz+1) [m/s]と中心de (nr,nz) [m²/s]を持つ状態。

        Returns
        -------
        ElectronTransport
            準備済みの自身。FiPyの係数、面のdiffusion_rates・boundary_ratesと、
            transport_limit・diffusion_limit [s]を更新する。密度と累積流出数は変更しない。

        Raises
        ------
        ValueError
            電子密度の正本・格子・電極マスク・面配列形状が初期化時と異なる、または気体のDが負・非有限。
        FloatingPointError
            面速度が非有限、または厳密なNumPy例外設定下で係数計算が失敗。

        Notes
        -----
        固定の面分類・距離は初期化時に作る。気体間Dは調和平均、吸収面Dは気体側値。
        片側Dが0の気体間面は拡散障壁となり、金属内のDは0として扱う。
        面係数D*A/dに外向き移流A*max(v_out,0)を加え、同じ境界係数を損失項と流出へ使う。
        合算上限はセルの移流・拡散流出対角の和、純拡散上限は拡散対角だけから求める。
        気体セルだけを評価し、両上限にCFL数・安全係数は掛けない。軸の面積は0。
        速度・Dを変更したら再度呼ぶ。密度のみが変わる試行・再試行では再利用できる。

        Examples
        --------
        >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
        >>> config = SimulationConfig(nr=2, nz=3, voltage=0., nmax=0.,
        ...                           background_ionization=0., t_end=0.)
        >>> state = initialize_simulation(config)
        >>> transport = ElectronTransport(state, config.delta).prepare(state)
        >>> transport.transport_limit, transport.diffusion_limit
        (inf, inf)
        >>> float(transport.boundary_rates.sum())
        0.0
        >>> losses = transport.step(state, config.dt, prepared=True)
        >>> float(losses.sum())
        0.0
        """
        grid = self.grid
        if (state.ne_var is not self.density or state.grid is not grid
                or state.ne.shape != self.shape or not np.array_equal(state.gas_mask, self.gas)):
            raise ValueError("transport density, grid and electrode mask must not change")
        speed = grid.pack_faces(state.vex, state.vey)
        if not np.all(np.isfinite(speed)):
            raise FloatingPointError("nonfinite transport velocity")
        coefficient = np.where(self.gas, state.de, 0.)
        if np.any(~np.isfinite(coefficient)) or np.any(coefficient < 0):
            raise ValueError("diffusion coefficients must be finite and nonnegative")
        low, high = grid.adjacent(coefficient)
        diffusion = low + high
        np.divide(2*low*high, low+high, out=diffusion, where=self.interior & (diffusion > 0.))
        self.diffusion_rates = diffusion*grid.area/self.distance
        self.boundary_rates = (grid.area*np.maximum(np.where(self.low_gas, speed, -speed), 0.)
                               + self.diffusion_rates)*self.boundary
        diffusion_loss = grid.sum_faces(self.diffusion_rates)
        rate = speed*grid.area
        loss = grid.sum_faces(np.maximum(rate, 0.), np.maximum(-rate, 0.)) + diffusion_loss
        maxima = (float(np.max((values/grid.volume)[self.gas])) for values in (loss, diffusion_loss))
        self.transport_limit, self.diffusion_limit = (1/value if value > 0. else math.inf for value in maxima)
        self.velocity.setValue(grid.directions*speed*self.interior)
        self.diffusion.setValue(diffusion*self.interior)
        self.absorption.setValue((grid.sum_faces(self.boundary_rates)/grid.volume).ravel(order='F'))
        return self

    def step(self, state, dt, *, prepared=False):
        """旧密度で電子輸送を1回更新し、境界ごとの流出数を返す。

        Parameters
        ----------
        state : SimulationState
            共有格子と計算配列。中心配列は(nr,nz)、面配列は方向別の全面形状。
            電極内の電子密度は0とする。共通時間ループではvalidate_densities()が検査する。
        dt : float
            実際の更新幅 [s]。有限かつ正。
        prepared : bool, optional
            Trueなら同じstateの速度・Dでprepare()済みの係数を使う。既定Falseでは再評価する。
            Trueで呼ぶ間は格子・電極形状・速度・Dを変更しない。

        Returns
        -------
        ndarray, shape (3,)
            流出電子数 [電極, 接地面, 上端＋外周]。2πを含む実粒子数。

        Raises
        ------
        ValueError
            dtが不正、面配列形状が不正、格子・電極マスクが変化、またはDが不正。
        FloatingPointError
            入力の密度・速度、または解・流出量が非有限。

        Notes
        -----
        neだけをその場で更新し、electron_lossesは変更しない。負密度のクリップは行わない。
        現在値はstate.ne_varと共有し、入力・出力間のコピーは行わない。
        毎試行の開始時にupdateOld()で旧値を確保し、求解例外・非有限解では旧値へ復元する。
        負密度を含む試行全体の棄却・7密度の復元と安定刻みの選択は呼出側が担当する。

        Examples
        --------
        >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
        >>> config = SimulationConfig(nr=2, nz=3, voltage=0., nmax=0.,
        ...                           background_ionization=0., t_end=0.)
        >>> state = initialize_simulation(config)
        >>> transport = ElectronTransport(state, config.delta)
        >>> losses = transport.step(state, config.dt)
        >>> float(state.ne.max()), float(state.electron_losses.sum())
        (0.0, 0.0)
        """
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("transport dt must be finite and positive")
        if not prepared:
            self.prepare(state)
        if not np.all(np.isfinite(state.ne)):
            raise FloatingPointError("nonfinite transport density")
        grid = self.grid
        self.density.updateOld()
        if not (np.any(self.velocity.value) or np.any(self.diffusion.value) or np.any(self.boundary_rates)):
            return np.zeros(3)
        old = self.density.old.value.reshape(self.shape, order='F')
        low_density, high_density = grid.adjacent(old)
        losses = 2*np.pi*dt*np.bincount(grid.boundary_kind,
            weights=self.boundary_rates*np.where(self.low_gas, low_density, high_density), minlength=3)
        try:
            self.equation.solve(var=self.density, dt=dt, solver=self.solver)
            if not np.all(np.isfinite(state.ne)) or not np.all(np.isfinite(losses)):
                raise FloatingPointError("nonfinite FiPy transport solution or boundary loss")
        except Exception:
            self.density.setValue(self.density.old.value)
            raise
        return losses
