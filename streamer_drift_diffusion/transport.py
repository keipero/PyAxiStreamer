"""FiPyによる円筒有限体積の陽的電子移流拡散と、吸収境界の粒子収支。"""
import numpy as np

from fipy import (CellVariable, ExplicitDiffusionTerm, ExplicitUpwindConvectionTerm,
                  FaceVariable, TransientTerm)
from fipy.solvers import LinearLUSolver



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
        各試行で同期する電子密度 [m⁻³]。旧値を保持する。
    velocity : fipy.FaceVariable
        気体間面の速度ベクトル [m/s]。
    diffusion : fipy.FaceVariable
        気体間面の拡散係数 [m²/s]。
    absorption : fipy.CellVariable
        セルの吸収損失係数 [s⁻¹]。
    gas : ndarray of bool, shape (nr, nz)
        初期の気体マスクのコピー。形状変更を検出する。

    Notes
    -----
    中心配列は(nr,nz)、vexは(nr+1,nz)、veyは(nr,nz+1)。deは等方拡散係数 [m²/s]。
    一次上流移流と拡散を同じ旧密度から陽的Eulerで更新する。
    気体間Dは調和平均。電極・外端は気体側Dと半セル距離による吸収で、外から供給しない。
    1シミュレーションにつき一度作り、格子と電極形状は計算中に変更しない。
    流出数の累積は受理を判断する呼出側が担当する。

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
        self.density = CellVariable(mesh=self.mesh, value=0., hasOld=True)
        self.velocity = FaceVariable(mesh=self.mesh, rank=1, value=0.)
        self.diffusion = FaceVariable(mesh=self.mesh, value=0.)
        self.absorption = CellVariable(mesh=self.mesh, value=0.)
        self.equation = (TransientTerm()
                         == ExplicitDiffusionTerm(coeff=self.diffusion)
                         - _ExplicitUpwind(coeff=self.velocity)
                         - self.absorption * self.density.old)
        self.solver = LinearLUSolver(tolerance=1e-12)

    def step(self, state, dt):
        """旧密度で電子輸送を1回更新し、境界ごとの流出数を返す。

        Parameters
        ----------
        state : SimulationState
            共有格子と計算配列。中心配列は(nr,nz)、面配列は方向別の全面形状。
        dt : float
            実際の更新幅 [s]。有限かつ正。

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
        現在値・旧値は毎試行の開始時にstateから再同期し、棄却後の再試行に対応する。
        数値異常はstate.neに解を反映する前に拒否する。安定刻みの選択は呼出側が担当する。

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
        if state.ne.shape != self.shape or not np.array_equal(state.gas_mask, self.gas):
            raise ValueError("transport grid and electrode mask must not change")
        if (not np.all(np.isfinite(state.ne)) or not np.all(np.isfinite(state.vex))
                or not np.all(np.isfinite(state.vey))):
            raise FloatingPointError("nonfinite transport density or velocity")

        grid = self.grid
        low_gas, high_gas = grid.adjacent(self.gas)
        interior = low_gas & high_gas
        speed = grid.pack_faces(state.vex, state.vey)
        diffusion = grid.diffusion_faces(state.de, self.gas)
        absorption = grid.area * (np.maximum(np.where(low_gas, speed, -speed), 0.)
                                  + diffusion/grid.distances(self.gas)) * (low_gas ^ high_gas)
        old = np.where(self.gas, state.ne, 0.)
        self.density.setValue(old.ravel(order='F'))
        self.density.updateOld()
        self.velocity.setValue(grid.directions * speed * interior)
        self.diffusion.setValue(diffusion * interior)
        self.absorption.setValue((grid.sum_faces(absorption)/grid.volume).ravel(order='F'))
        self.equation.solve(var=self.density, dt=dt, solver=self.solver)
        result = np.asarray(self.density.value).reshape(self.shape, order='F')
        low_density, high_density = grid.adjacent(old)
        losses = 2*np.pi*dt*np.bincount(grid.boundary_kind,
            weights=absorption*np.where(low_gas, low_density, high_density), minlength=3)
        if not np.all(np.isfinite(result)) or not np.all(np.isfinite(losses)):
            raise FloatingPointError("nonfinite FiPy transport solution or boundary loss")
        state.ne[:] = result
        return losses
