"""共有の円筒格子、全面の隣接関係・幾何係数・境界分類と有限体積行列。"""
import numpy as np
from scipy.sparse import coo_matrix


class CylindricalGrid:
    """軸対称r-zの等間隔構造格子と、全処理で共有する面情報を生成する。

    Parameters
    ----------
    config : SimulationConfig
        格子寸法・物理定数・境界・刻み制御を保持する設定。変更しない。

    Attributes
    ----------
    shape : tuple of int
        セル配列形状(nr,nz)。
    n_cells, n_faces : int
        セル数と全面数。
    spacing : float
        両方向共通のセル幅 [m]。
    volume : ndarray, shape (nr, nz)
        2πを省略したセル体積 [m³]。
    area : ndarray, shape (nfaces,)
        2πを省略した全面積 [m²]。対称軸は0。
    areas : tuple of ndarray
        全半径面(nr+1,nz)と全軸方向面(nr,nz+1)の面積。
    low, high : ndarray of int, shape (nfaces,)
        各面の座標が小さい側・大きい側のセル番号。領域外は-1。
    valid_low, valid_high : ndarray of bool, shape (nfaces,)
        対応する隣接セルが領域内に存在するか。
    directions : ndarray, shape (2, nfaces)
        面が属する座標方向の単位ベクトル。符号は常に座標増加方向。
    face_ids : tuple of ndarray of int
        半径・軸方向の全面配列から面番号への対応表。
    boundary_kind : ndarray of int, shape (nfaces,)
        内部面0、接地面1、その他外部面2。実際の吸収面は気体マスクで別途判定する。
    r, z : ndarray
        各方向のセル下側境界 [m]。最終境界を含めない。
    rh, zh : ndarray
        各方向のセル中心 [m]。

    Notes
    -----
    セル配列は(nr,nz)で、セル番号i+nr*jはFortran順に対応する。
    面番号はFiPyのGrid2Dと同じで、軸方向面i+nr*jの後に半径面i+(nr+1)*jが続く。
    面値の正方向は座標増加方向。low/highがその両側セルで、領域外は-1。
    面積・体積は共通係数2πを省略し、粒子数・電荷の集計時に一度だけ掛ける。
    格子形状は計算中に変更しない。気体・電極の区別は各操作のgas引数で与える。

    Examples
    --------
    >>> from streamer_drift_diffusion import SimulationConfig
    >>> grid = CylindricalGrid(SimulationConfig(nr=2, nz=3))
    >>> grid.volume.shape
    (2, 3)
    >>> [area.shape for area in grid.areas]
    [(3, 3), (2, 4)]
    """
    def __init__(self, config):
        """設定から共有メッシュと面の隣接・方向・幾何情報を初期化する。

        Parameters
        ----------
        config : SimulationConfig
            格子寸法・物理定数・境界・刻み制御を保持する設定。変更しない。

        Notes
        -----
        configはvalidate_config()による検査後に渡す。属性の単位・形状はクラスの説明を参照。

        Examples
        --------
        >>> from streamer_drift_diffusion import SimulationConfig
        >>> grid = CylindricalGrid(SimulationConfig(nr=2, nz=3))
        >>> grid.shape
        (2, 3)
        """
        nr, nz, d = config.nr, config.nz, config.delta
        self.shape, self.spacing = (nr, nz), d
        self.n_cells, n_axial = nr*nz, nr*(nz+1)
        self.n_faces = n_axial + (nr+1)*nz
        self.r, self.z = np.arange(nr)*d, np.arange(nz)*d
        self.rh, self.zh = (np.arange(nr)+.5)*d, (np.arange(nz)+.5)*d
        self.volume = np.repeat(self.rh[:, None]*(d*d), nz, axis=1)
        i, j = np.arange(nr+1)[:, None], np.arange(nz)[None, :]
        radial = n_axial + i + (nr+1)*j
        k, m = np.arange(nr)[:, None], np.arange(nz+1)[None, :]
        axial = k + nr*m
        self.face_ids = (radial, axial)
        cell = lambda a, b, inside: np.where(inside, a + nr*b, -1)
        self.low = np.empty(self.n_faces, dtype=int)
        self.high = np.empty(self.n_faces, dtype=int)
        self.low[radial], self.high[radial] = cell(i-1, j, i > 0), cell(i, j, i < nr)
        self.low[axial], self.high[axial] = cell(k, m-1, m > 0), cell(k, m, m < nz)
        self.valid_low, self.valid_high = self.low >= 0, self.high >= 0
        self.directions = np.zeros((2, self.n_faces))
        self.directions[0, radial], self.directions[1, axial] = 1., 1.
        self.area = np.empty(self.n_faces)
        self.area[radial], self.area[axial] = np.broadcast_to(i*d*d, radial.shape), self.rh[:, None]*d
        exterior = ~(self.valid_low & self.valid_high)
        self.boundary_kind = np.where(exterior, 2, 0)
        self.boundary_kind[axial[:, 0]] = 1
        self.areas = self.split_faces(self.area)

    def pack_faces(self, radial, axial):
        """半径・軸方向の全面配列を、共有メッシュの面番号順へ変換する。

        Parameters
        ----------
        radial : ndarray, shape (nr+1, nz)
            座標増加方向を正とする半径面の値。
        axial : ndarray, shape (nr, nz+1)
            同じ単位・意味を持つ軸方向面の値。

        Returns
        -------
        ndarray, shape (nfaces,)
            面番号順の新しい配列。dtypeは両入力から決める。

        Raises
        ------
        ValueError
            片方の面配列の形状が共有格子に一致しない。

        Examples
        --------
        >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
        >>> config = SimulationConfig(nr=2, nz=3, voltage=0., nmax=0.,
        ...                           background_ionization=0., t_end=0.)
        >>> state = initialize_simulation(config)
        >>> grid = state.grid
        >>> import numpy as np
        >>> values = grid.pack_faces(np.zeros((3, 3)), np.ones((2, 4)))
        >>> values.shape
        (17,)
        """
        if radial.shape != self.face_ids[0].shape or axial.shape != self.face_ids[1].shape:
            raise ValueError("velocities must have radial (nr+1,nz) and axial (nr,nz+1) face shapes")
        result = np.empty(self.n_faces, dtype=np.result_type(radial, axial))
        result[self.face_ids[0]], result[self.face_ids[1]] = radial, axial
        return result

    def split_faces(self, values):
        """共有メッシュ順の値を、全半径面・全軸方向面へ分ける。

        Parameters
        ----------
        values : array_like, shape (nfaces,)
            面番号順の値。単位は呼出側の量に従う。

        Returns
        -------
        tuple of ndarray
            形状(nr+1,nz)と(nr,nz+1)の配列。入力値のコピー。

        Notes
        -----
        pack_faces()と逆の対応を使う。境界面を省略しない。

        Examples
        --------
        >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
        >>> config = SimulationConfig(nr=2, nz=3, voltage=0., nmax=0.,
        ...                           background_ionization=0., t_end=0.)
        >>> state = initialize_simulation(config)
        >>> grid = state.grid
        >>> import numpy as np
        >>> radial, axial = grid.split_faces(np.arange(grid.n_faces))
        >>> radial.shape, axial.shape
        ((3, 3), (2, 4))
        """
        return tuple(np.asarray(values)[ids] for ids in self.face_ids)

    def adjacent(self, values):
        """各面の両側のセル値を取得し、領域外には0を与える。

        Parameters
        ----------
        values : ndarray, shape (nr, nz)
            セル中心値。bool配列の場合、領域外はFalse。

        Returns
        -------
        tuple of ndarray
            各々形状(nfaces,)の配列。入力のdtypeと単位を保持する。

        Examples
        --------
        >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
        >>> config = SimulationConfig(nr=2, nz=3, voltage=0., nmax=0.,
        ...                           background_ionization=0., t_end=0.)
        >>> state = initialize_simulation(config)
        >>> grid = state.grid
        >>> import numpy as np
        >>> low, high = grid.adjacent(np.ones(grid.shape))
        >>> low.shape == high.shape
        True
        """
        padded = np.append(values.ravel(order='F'), np.zeros(1, dtype=values.dtype))
        return padded[self.low], padded[self.high]

    def interior(self, gas):
        """両側が気体セルである内部面を判定する。

        Parameters
        ----------
        gas : ndarray of bool, shape (nr, nz)
            気体True、電極Falseのマスク。

        Returns
        -------
        ndarray of bool, shape (nfaces,)
            気体－気体面だけがTrue。外部境界・電極面はFalse。

        Examples
        --------
        >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
        >>> config = SimulationConfig(nr=2, nz=3, voltage=0., nmax=0.,
        ...                           background_ionization=0., t_end=0.)
        >>> state = initialize_simulation(config)
        >>> grid = state.grid
        >>> bool(grid.interior(state.gas_mask).any())
        True
        """
        low, high = self.adjacent(gas)
        return low & high

    def distances(self, gas):
        """面の両側が気体かどうかに応じて離散化距離を返す。

        Parameters
        ----------
        gas : ndarray of bool, shape (nr, nz)
            気体・電極マスク。

        Returns
        -------
        ndarray, shape (nfaces,)
            気体間はspacing、それ以外はspacing/2 [m]。

        Notes
        -----
        気体－電極・外部境界では、気体中心から境界面までの半セル距離になる。
        金属間や対称軸にも補助値spacing/2を返すが、物理流束はマスク・面積0で無効にする。

        Examples
        --------
        >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
        >>> config = SimulationConfig(nr=2, nz=3, voltage=0., nmax=0.,
        ...                           background_ionization=0., t_end=0.)
        >>> state = initialize_simulation(config)
        >>> grid = state.grid
        >>> import numpy as np
        >>> distances = grid.distances(state.gas_mask)
        >>> bool(np.allclose(distances[grid.interior(state.gas_mask)], config.delta))
        True
        """
        return np.where(self.interior(gas), self.spacing, self.spacing/2)


    def sum_faces(self, low_values, high_values=None):
        """面の寄与を、隣接する両側のセルへ集計する。

        Parameters
        ----------
        low_values : ndarray, shape (nfaces,)
            面の座標が小さい側のセルへ加える値。
        high_values : ndarray, shape (nfaces,), optional
            大きい側へ加える値。Noneならlow_valuesと同じ値を加える。

        Returns
        -------
        ndarray, shape (nr, nz)
            集計後のセル値。単位は入力と同じ。

        Notes
        -----
        領域外セルへの寄与は無視する。気体マスクによる除外や、面積の乗算は行わない。
        保存則の発散評価では両側に逆符号を、損失対角では同符号の寄与を渡す。

        Examples
        --------
        >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
        >>> config = SimulationConfig(nr=2, nz=3, voltage=0., nmax=0.,
        ...                           background_ionization=0., t_end=0.)
        >>> state = initialize_simulation(config)
        >>> grid = state.grid
        >>> import numpy as np
        >>> float(grid.sum_faces(np.ones(grid.n_faces)).max())
        4.0
        """
        high_values = low_values if high_values is None else high_values
        low = np.bincount(self.low[self.valid_low], weights=low_values[self.valid_low], minlength=self.n_cells)
        high = np.bincount(self.high[self.valid_high], weights=high_values[self.valid_high], minlength=self.n_cells)
        return (low+high).reshape(self.shape, order='F')

    def conductance_matrix(self, conductance, diagonal, cells=None):
        """面コンダクタンスとセル対角から、対称な有限体積行列を組む。

        Parameters
        ----------
        conductance : ndarray, shape (nfaces,)
            内部面の結合係数（例: 面積/距離）。0の面と外部面は結合しない。
        diagonal : ndarray, shape (ncells,)
            対象セルの対角への追加値。cells指定時はその対象セルだけの長さ。
            Dirichlet境界・吸収・体積項など、面結合以外の寄与を与える。
        cells : ndarray of bool, shape (nr, nz), optional
            未知数にするセル。Noneなら全セル。行列の行はFortran順に詰める。

        Returns
        -------
        scipy.sparse.csc_matrix
            対角に結合係数の和とdiagonal、非対角に-conductanceを持つ行列。
            -∇·(c∇u)の体積積分形で、splu()へそのまま渡せる。

        Raises
        ------
        ValueError
            結合する面が対象外のセルに接する、またはdiagonalの長さが不正。

        Notes
        -----
        2πは省略した面積・体積の規約に従う。Dirichlet面の寄与は呼出側がdiagonalと右辺へ入れる。

        Examples
        --------
        >>> from streamer_drift_diffusion import SimulationConfig
        >>> import numpy as np
        >>> grid = CylindricalGrid(SimulationConfig(nr=2, nz=1))
        >>> conductance = np.where(grid.valid_low & grid.valid_high, 1., 0.)
        >>> grid.conductance_matrix(conductance, np.zeros(2)).toarray().tolist()
        [[1.0, -1.0], [-1.0, 1.0]]
        """
        flat = np.ones(self.n_cells, dtype=bool) if cells is None else np.asarray(cells).ravel(order='F')
        index = np.full(self.n_cells, -1)
        index[flat] = np.arange(np.count_nonzero(flat))
        size = np.count_nonzero(flat)
        if np.shape(diagonal) != (size,):
            raise ValueError("diagonal must have one value per unknown cell")
        faces = np.flatnonzero(self.valid_low & self.valid_high & (conductance != 0.))
        left, right = index[self.low[faces]], index[self.high[faces]]
        if np.any(left < 0) or np.any(right < 0):
            raise ValueError("coupled faces must connect unknown cells")
        values, cell = conductance[faces], np.arange(size)
        return coo_matrix(
            (np.r_[-values, -values, values, values, diagonal],
             (np.r_[left, right, left, right, cell], np.r_[right, left, left, right, cell])),
            shape=(size, size)).tocsc()


def initialize_grid(config):
    """共有格子と、既存の計算・出力が参照する幾何配列を返す。

    Parameters
    ----------
    config : SimulationConfig
        格子寸法・物理定数・境界・刻み制御を保持する設定。変更しない。

    Returns
    -------
    dict
        grid、r/z、rh/zh、dr/dz、volume、sr/sz、gas_mask、electron_lossesを含む。
        中心配列は(nr,nz)、座標は各方向の1次元。electron_lossesは長さ3。

    Notes
    -----
    幾何量はCylindricalGridから取得し、volume/sr/szはその配列への参照・ビュー。
    気体マスクは双曲線電極の階段近似で、累積流出数は0から始める。
    configの検査は呼出側のvalidate_config()が担当する。

    Examples
    --------
    >>> from streamer_drift_diffusion import SimulationConfig
    >>> arrays = initialize_grid(SimulationConfig(nr=2, nz=3))
    >>> arrays['volume'] is arrays['grid'].volume
    True
    >>> arrays['electron_losses'].tolist()
    [0.0, 0.0, 0.0]
    """
    grid = CylindricalGrid(config)
    gas = (grid.zh[None, :]/config.electrode_gap)**2 - grid.rh[:, None]**2 / (
        config.cathode_radius*config.electrode_gap) < 1.
    return dict(grid=grid, r=grid.r, z=grid.z, rh=grid.rh, zh=grid.zh,
                dr=np.full(config.nr, config.delta), dz=np.full(config.nz, config.delta),
                volume=grid.volume, sr=grid.areas[0][:-1], sz=grid.areas[1][:, :-1],
                gas_mask=gas, electron_losses=np.zeros(3))
