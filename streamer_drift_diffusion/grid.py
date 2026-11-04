"""共有の円筒メッシュ、全面の隣接関係・幾何係数・境界分類。"""
import numpy as np
from fipy import CylindricalGrid2D, serialComm


class CylindricalGrid:
    """軸対称r-zのFiPyメッシュと、全処理で共有する面情報を生成する。

    Parameters
    ----------
    config : SimulationConfig
        格子寸法・物理定数・境界・刻み制御を保持する設定。変更しない。

    Attributes
    ----------
    mesh : fipy.meshes.mesh.Mesh
        一度だけ生成するCylindricalGrid2D。
    shape : tuple of int
        セル配列形状(nr,nz)。
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
        半径・軸方向の全面配列からFiPy面番号への対応表。
    boundary_kind : ndarray of int, shape (nfaces,)
        内部面0、接地面1、その他外部面2。実際の吸収面は気体マスクで別途判定する。
    r, z : ndarray
        各方向のセル下側境界 [m]。最終境界を含めない。
    rh, zh : ndarray
        各方向のセル中心 [m]。

    Notes
    -----
    FiPyのx=r、y=z。セル配列は(nr,nz)で、FiPyセル番号にはFortran順で対応する。
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
        self.shape, self.spacing = (config.nr, config.nz), config.delta
        self.mesh = CylindricalGrid2D(nr=config.nr, nz=config.nz, dr=config.delta,
                                      dz=config.delta, communicator=serialComm)
        centers = np.asarray(self.mesh.cellCenters).reshape((2, *self.shape), order='F')
        self.rh, self.zh = centers[0, :, 0], centers[1, 0, :]
        self.volume = np.asarray(self.mesh.cellVolumes).reshape(self.shape, order='F')
        normals = np.asarray(self.mesh.faceNormals)
        self.directions = np.abs(normals)
        ids = np.ma.filled(self.mesh.faceCellIDs, -1)
        negative = normals.sum(axis=0) < 0
        self.low = np.where(negative, ids[1], ids[0])
        self.high = np.where(negative, ids[0], ids[1])
        self.valid_low, self.valid_high = self.low >= 0, self.high >= 0
        r, z = np.asarray(self.mesh.faceCenters)
        self.area = r * self.spacing
        self.boundary_kind = np.where(self.mesh.exteriorFaces, np.where(z == 0., 1, 2), 0)
        self.face_ids = (np.empty((config.nr+1, config.nz), dtype=int),
                         np.empty((config.nr, config.nz+1), dtype=int))
        for axis, mapping in enumerate(self.face_ids):
            face = np.flatnonzero(self.directions[axis])
            coords = np.asarray(self.mesh.faceCenters)[:, face]/self.spacing
            indices = np.rint(coords - .5*(1-self.directions[:, face])).astype(int)
            mapping[indices[0], indices[1]] = face
        self.areas = self.split_faces(self.area)
        self.r = r[self.face_ids[0][:-1, 0]]
        self.z = z[self.face_ids[1][0, :-1]]

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
        result = np.empty(self.mesh.numberOfFaces, dtype=np.result_type(radial, axial))
        result[self.face_ids[0]], result[self.face_ids[1]] = radial, axial
        return result

    def split_faces(self, values):
        """共有メッシュ順の値を、全半径面・全軸方向面へ分ける。

        Parameters
        ----------
        values : array_like, shape (nfaces,)
            FiPy面番号順の値。単位は呼出側の量に従う。

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
        >>> radial, axial = grid.split_faces(np.arange(grid.mesh.numberOfFaces))
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

    def diffusion_faces(self, coefficient, gas):
        """中心の拡散係数から、気体間・吸収面に用いる面係数を求める。

        Parameters
        ----------
        coefficient : float or ndarray, shape (nr, nz)
            等方拡散係数 [m²/s]。気体セルでは有限かつ非負。
        gas : ndarray of bool, shape (nr, nz)
            気体True、電極False。

        Returns
        -------
        ndarray, shape (nfaces,)
            面の拡散係数 [m²/s]。金属間は0。

        Raises
        ------
        ValueError
            気体セルの係数が負または非有限。

        Notes
        -----
        気体間は調和平均2*D_low*D_high/(D_low+D_high)。片側が0なら面係数も0。
        吸収面は気体側のDを使用し、半セル距離はdistances()で別途反映する。

        Examples
        --------
        >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
        >>> config = SimulationConfig(nr=2, nz=3, voltage=0., nmax=0.,
        ...                           background_ionization=0., t_end=0.)
        >>> state = initialize_simulation(config)
        >>> grid = state.grid
        >>> float(grid.diffusion_faces(0., state.gas_mask).max())
        0.0
        """
        coefficient = np.where(gas, coefficient, 0.)
        if np.any(~np.isfinite(coefficient)) or np.any(coefficient < 0):
            raise ValueError("diffusion coefficients must be finite and nonnegative")
        low, high = self.adjacent(coefficient)
        result = low + high
        np.divide(2*low*high, low+high, out=result,
                  where=self.interior(gas) & (result > 0.))
        return result

    def diffusion_rates(self, coefficient, gas):
        """面の拡散コンダクタンスD×面積/距離を求める。

        Parameters
        ----------
        coefficient : float or ndarray, shape (nr, nz)
            中心の拡散係数 [m²/s]。
        gas : ndarray of bool, shape (nr, nz)
            気体・電極マスク。

        Returns
        -------
        ndarray, shape (nfaces,)
            2πを省略したD*A/d [m³/s]。対称軸は0。

        Raises
        ------
        ValueError
            気体側の拡散係数が負または非有限。

        Notes
        -----
        セルへの損失対角集計にはsum_faces()を使う。係数平均と境界距離は共通の幾何情報に従う。

        Examples
        --------
        >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
        >>> config = SimulationConfig(nr=2, nz=3, voltage=0., nmax=0.,
        ...                           background_ionization=0., t_end=0.)
        >>> state = initialize_simulation(config)
        >>> grid = state.grid
        >>> import numpy as np
        >>> rates = grid.diffusion_rates(0., state.gas_mask)
        >>> bool(np.all(rates == 0.))
        True
        """
        return self.diffusion_faces(coefficient, gas)*self.area/self.distances(gas)

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
        >>> float(grid.sum_faces(np.ones(grid.mesh.numberOfFaces)).max())
        4.0
        """
        high_values = low_values if high_values is None else high_values
        low = np.bincount(self.low[self.valid_low], weights=low_values[self.valid_low], minlength=self.mesh.numberOfCells)
        high = np.bincount(self.high[self.valid_high], weights=high_values[self.valid_high], minlength=self.mesh.numberOfCells)
        return (low+high).reshape(self.shape, order='F')


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
