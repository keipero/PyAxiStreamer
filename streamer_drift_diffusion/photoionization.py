"""乾燥N₂/O₂のZheleznyak光電離源を3項Helmholtz近似で求める。"""
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu


# Bourdon et al. (2007), Table 2, doi:10.1088/0963-0252/16/3/026.
# 元の係数はそれぞれ cm^-2 Torr^-2、cm^-1 Torr^-1。
_AMPLITUDES = np.array([1.986e-4, 0.0051, 0.4886]) * 1e4
_ABSORPTION = np.array([0.0553, 0.1460, 0.89]) * 100


class PhotoionizationSolver:
    """共有格子で光電離源を求め、3個のLU分解を再利用する。

    Parameters
    ----------
    grid : CylindricalGrid
        面積と体積を共有する軸対称格子。
    gas : ndarray of bool
        気体セルのマスク。形状はgrid.shape。
    config : SimulationConfig
        酸素モル分率と無次元の光電離効率。

    Attributes
    ----------
    rate : ndarray
        直近の光電離源 S_ph [m^-3 s^-1]。電極セルでは0。

    Notes
    -----
    各項について (-∇² + κ²) S = A p_O2² I を解く。
    p_O2は1 atmと初期酸素モル分率で固定し、発光源Iには衝突消光を含める。
    軸は対称、電極と接地面ではS=0、外周・上端の開放面では
    ∂S/∂n + κ S = 0 とする。Robin条件は外部への指数減衰を仮定した近似。
    行列は最初の非ゼロ源で分解し、以後は右辺のみを更新する。

    Examples
    --------
    >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
    >>> state = initialize_simulation(SimulationConfig(nr=2, nz=3))
    >>> state.photoionization.solve(np.zeros(state.grid.shape)).max().item()
    0.0
    """

    def __init__(self, grid, gas, config):
        self.rate = np.zeros(grid.shape, order="F")
        self._grid = grid
        self._gas = gas
        self._mask = gas.ravel(order="F")
        self._volume = grid.volume.ravel(order="F")[self._mask]
        oxygen_torr = 760. * config.oxygen_fraction
        self._absorption = _ABSORPTION * oxygen_torr
        self._weights = _AMPLITUDES * oxygen_torr**2
        # 40 mbarの消光圧を用いる。全圧は既存の気体密度モデルと同じ1 atm。
        self._yield = config.photoionization_efficiency * 4000. / (101325. + 4000.)
        self._factors = ()

    def _factorize(self):
        """気体セルの有限体積行列を組み立て、各吸収項のLU分解を返す。

        Returns
        -------
        tuple of scipy.sparse.linalg.SuperLU
            Bourdonの表の順に並んだ3個のLU分解。
        """
        grid = self._grid
        low_gas, high_gas = grid.adjacent(self._gas)
        interior = low_gas & high_gas
        boundary = low_gas ^ high_gas
        open_faces = boundary & (grid.boundary_kind == 2)
        conductance = np.where(interior | (boundary & ~open_faces),
                               grid.area / grid.distances(self._gas), 0.)
        diagonal = grid.sum_faces(conductance * low_gas, conductance * high_gas)
        open_area = grid.area * open_faces
        surface = grid.sum_faces(open_area * low_gas, open_area * high_gas)
        surface = surface.ravel(order="F")[self._mask]
        cell = np.cumsum(self._mask) - 1
        left, right = cell[grid.low[interior]], cell[grid.high[interior]]
        indices = np.arange(self._volume.size)
        edges = -conductance[interior]
        negative_laplacian = coo_matrix(
            (np.r_[edges, edges, diagonal.ravel(order="F")[self._mask]],
             (np.r_[left, right, indices], np.r_[right, left, indices])),
            shape=(indices.size, indices.size),
        ).tocsc()
        half_cell = grid.spacing / 2
        return tuple(splu(negative_laplacian + diags(
            kappa**2 * self._volume + surface * kappa / (1. + kappa * half_cell),
            format="csc",
        )) for kappa in self._absorption)

    def solve(self, impact, *, return_components=False):
        """電子衝突電離源から光電離源を更新する。

        Parameters
        ----------
        impact : ndarray
            N₂・O₂の電子衝突電離の和 [m^-3 s^-1]。形状はgrid.shape。
        return_components : bool, optional
            Trueなら検証用に3項の分布を個別に返す。既定False。

        Returns
        -------
        ndarray or tuple of ndarray
            更新したrateへの参照。return_components=Trueでは
            (rate, components)を返す。componentsの形状は(3, nr, nz)で、
            Bourdonの表2の順。単位はいずれも[m^-3 s^-1]。

        Raises
        ------
        ValueError
            入力の形状が共有格子と異なる。
        RuntimeError
            気体セルの入力または解が非有限か負。

        Examples
        --------
        既に用意したソルバで局所的な衝突電離から光電離源を求める::

            impact = np.zeros(state.grid.shape)
            impact[0, 0] = 1e24
            rate = state.photoionization.solve(impact)
            rate, components = state.photoionization.solve(impact, return_components=True)
        """
        if impact.shape != self.rate.shape:
            raise ValueError("impact ionization source shape does not match the grid")
        source = impact.ravel(order="F")[self._mask]
        if not np.all(np.isfinite(source)) or np.any(source < 0):
            raise RuntimeError("invalid impact ionization source for photoionization")
        if not self._yield or not np.any(source):
            self.rate.fill(0.)
            if return_components:
                return self.rate, np.zeros((len(self._weights), *self.rate.shape))
            return self.rate
        if not self._factors:
            self._factors = self._factorize()
        rhs = self._yield * self._volume * source
        terms = (weight * factor.solve(rhs)
                 for weight, factor in zip(self._weights, self._factors))
        if return_components:
            terms = tuple(terms)
        values = sum(terms)
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise RuntimeError("invalid Helmholtz photoionization solution")
        self.rate.ravel(order="F")[self._mask] = values
        if return_components:
            components = np.zeros((len(self._weights), *self.rate.shape))
            for component, term in zip(components, terms):
                component.T[self._gas.T] = term
            return self.rate, components
        return self.rate

    def close(self):
        """LU分解への参照を解放し、直近の光電離源を保持する。

        Examples
        --------
        計算終了後に分解を解放する::

            state.photoionization.close()
        """
        self._factors = ()
