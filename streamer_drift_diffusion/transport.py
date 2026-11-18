"""円筒有限体積の陽的電子移流拡散と、吸収境界の粒子収支。"""
import math
import numpy as np

from ._transport_kernel import prepare_coefficients, transport_update


def _centered_stencil(gas):
    """各方向で中心と両隣が気体にあるセルを返す。"""
    radial = np.zeros_like(gas)
    axial = np.zeros_like(gas)
    radial[1:-1] = gas[:-2] & gas[1:-1] & gas[2:]
    axial[:, 1:-1] = gas[:, :-2] & gas[:, 1:-1] & gas[:, 2:]
    return np.stack((radial.ravel(order="F"), axial.ravel(order="F")))


class ElectronTransport:
    """共有格子上の電子移流拡散と、吸収面の流出数を計算する。

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
    grid : CylindricalGrid
        state.gridへの参照。
    density : ndarray, shape (nr, nz)
        state.neと同一の電子密度配列 [m⁻³]。
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
    Van Leer（MC制限）移流と拡散を同じ旧密度からNumbaで陽的に更新する。
    FiPyのVanLeerConvectionTermと同じ再構築式で、検証用のFiPy参照実装はtests/fipy_reference.pyにある。
    気体間Dは調和平均。電極・外端は気体側Dと半セル距離による吸収で、外から供給しない。
    1シミュレーションにつき一度作り、格子と電極形状は計算中に変更しない。
    流出数の累積は受理を判断する呼出側が担当する。
    時間ループはprepare()の面係数をCFL・輸送・流出数へ共用する。
    単独のstep()は既定で係数を再評価し、速度・Dの変更を反映する。

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
        """共有格子から、固定の面分類・距離・境界写しの対応を準備する。

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
        >>> transport.density is state.ne
        True
        """
        self.grid = state.grid
        if spacing != self.grid.spacing:
            raise ValueError("transport spacing must match the shared grid")
        self.shape = self.grid.shape
        self.gas = state.gas_mask.copy()
        self.low_gas, self.high_gas = self.grid.adjacent(self.gas)
        self.interior = self.low_gas & self.high_gas
        self.boundary = self.low_gas ^ self.high_gas
        self.distance = self.grid.distances(self.gas)
        self.density = state.ne
        grid = self.grid
        self._gas_flat = self.gas.ravel(order="F")
        self._centered = _centered_stencil(self.gas)
        self._volume = grid.volume.ravel(order="F")
        self._boundary_sources = np.full(grid.area.size, -1, dtype=int)
        self._coefficient_area = grid.area.copy()
        if state.poisson.flux_neumann_boundaries:
            self._coefficient_area[self.boundary] = 0.
            radial, axial = grid.face_ids
            for lower, upper in ((radial[:-1], radial[1:]), (axial[:, :-1], axial[:, 1:])):
                for face, source in ((lower, upper), (upper, lower)):
                    copy = self.boundary[face] & self.interior[source] & (grid.area[face] > 0.)
                    self._boundary_sources[face[copy]] = source[copy]
        self._copy_faces = np.flatnonzero(self._boundary_sources >= 0)
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
        準備済みの自身。面の移流・拡散・境界流出係数と、
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
        合算上限はセルの移流・拡散流出の上界から求める。純拡散上限は拡散対角だけを使う。
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
        if (state.ne is not self.density or state.grid is not grid
                or state.ne.shape != self.shape or not np.array_equal(state.gas_mask, self.gas)):
            raise ValueError("transport density, grid and electrode mask must not change")
        speed = grid.pack_faces(state.vex, state.vey)
        if not np.all(np.isfinite(speed)):
            raise FloatingPointError("nonfinite transport velocity")
        coefficient = np.where(self.gas, state.de, 0.)
        if np.any(~np.isfinite(coefficient)) or np.any(coefficient < 0):
            raise ValueError("diffusion coefficients must be finite and nonnegative")
        (self.advection_rates, self.diffusion_rates, self.boundary_rates,
         transport_max, diffusion_max) = prepare_coefficients(
            speed, coefficient.ravel(order="F"), self._gas_flat, *grid.face_ids,
            grid.low, grid.high, self.low_gas, self.high_gas, self.interior,
            self._coefficient_area, self._volume, self.distance,
        )
        self.transport_limit = 1./transport_max if transport_max > 0. else math.inf
        self.diffusion_limit = 1./diffusion_max if diffusion_max > 0. else math.inf
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
        有限性を検査してから結果を反映するため、失敗時のneは更新前のまま。
        負密度を含む試行全体の棄却・密度の復元と安定刻みの選択は呼出側が担当する。

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
        new, losses = transport_update(
            state.ne.ravel(order="F"), self._gas_flat, self._centered,
            *grid.face_ids, grid.low, grid.high, self.low_gas, self.interior,
            grid.area, self._volume, self.advection_rates, self.diffusion_rates,
            self.boundary_rates, grid.boundary_kind, self._boundary_sources, grid.spacing, dt,
        )
        if not np.all(np.isfinite(new)) or not np.all(np.isfinite(losses)):
            raise FloatingPointError("nonfinite Numba transport solution or boundary loss")
        state.ne[:] = new.reshape(self.shape, order="F")
        return losses
