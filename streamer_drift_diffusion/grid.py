"""共有の円筒メッシュ、全面の隣接関係・幾何係数・境界分類。"""
import numpy as np
from fipy import CylindricalGrid2D, serialComm


class CylindricalGrid:
    """x=r、y=z。セル配列は(nr,nz)、FiPyセル番号はFortran順。

    面値は座標増加方向を正とし、low/highは両側のセル番号（領域外は-1）。
    面積・体積は2πを省略する。格子は計算中に変更しない。
    """
    def __init__(self, config):
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
        """全半径面・全軸方向面を共有メッシュの面番号へ変換する。"""
        if radial.shape != self.face_ids[0].shape or axial.shape != self.face_ids[1].shape:
            raise ValueError("velocities must have radial (nr+1,nz) and axial (nr,nz+1) face shapes")
        result = np.empty(self.mesh.numberOfFaces, dtype=np.result_type(radial, axial))
        result[self.face_ids[0]], result[self.face_ids[1]] = radial, axial
        return result

    def split_faces(self, values):
        return tuple(np.asarray(values)[ids] for ids in self.face_ids)

    def adjacent(self, values):
        """各面の両側セル値。領域外は0（boolならFalse）。"""
        padded = np.append(values.ravel(order='F'), np.zeros(1, dtype=values.dtype))
        return padded[self.low], padded[self.high]

    def interior(self, gas):
        low, high = self.adjacent(gas)
        return low & high

    def distances(self, gas):
        """気体間は全セル幅、吸収・Dirichlet面までは半セル幅。"""
        return np.where(self.interior(gas), self.spacing, self.spacing/2)

    def diffusion_faces(self, coefficient, gas):
        """気体間は調和平均、吸収面は気体側D。金属間は0。"""
        coefficient = np.where(gas, coefficient, 0.)
        if np.any(~np.isfinite(coefficient)) or np.any(coefficient < 0):
            raise ValueError("diffusion coefficients must be finite and nonnegative")
        low, high = self.adjacent(coefficient)
        result = low + high
        np.divide(2*low*high, low+high, out=result,
                  where=self.interior(gas) & (result > 0.))
        return result

    def diffusion_rates(self, coefficient, gas):
        return self.diffusion_faces(coefficient, gas)*self.area/self.distances(gas)

    def sum_faces(self, low_values, high_values=None):
        """面の寄与を両側セルへ集計する。方向別の異なる寄与も扱う。"""
        high_values = low_values if high_values is None else high_values
        low = np.bincount(self.low[self.valid_low], weights=low_values[self.valid_low], minlength=self.mesh.numberOfCells)
        high = np.bincount(self.high[self.valid_high], weights=high_values[self.valid_high], minlength=self.mesh.numberOfCells)
        return (low+high).reshape(self.shape, order='F')


def initialize_grid(config):
    """共有格子と幾何配列への参照。既存のPoisson・出力も同じ配列を使う。"""
    grid = CylindricalGrid(config)
    gas = (grid.zh[None, :]/config.electrode_gap)**2 - grid.rh[:, None]**2 / (
        config.cathode_radius*config.electrode_gap) < 1.
    return dict(grid=grid, r=grid.r, z=grid.z, rh=grid.rh, zh=grid.zh,
                dr=np.full(config.nr, config.delta), dz=np.full(config.nz, config.delta),
                volume=grid.volume, sr=grid.areas[0][:-1], sz=grid.areas[1][:, :-1],
                gas_mask=gas, electron_losses=np.zeros(3))
