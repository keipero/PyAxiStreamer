"""検証専用のFiPy参照実装。本体はFiPyに依存しない。

FiPyの標準項で同じ有限体積式を組み、NumPy・Numba・SciPy疎行列の本体と比較する。
FiPyはテスト依存（``pip install ".[test]"``）で、本体の実行には不要。
"""
import math
import os

os.environ.setdefault("FIPY_SOLVERS", "scipy")

import numpy as np
from fipy import (CellVariable, CylindricalGrid2D, DiffusionTerm, ExplicitDiffusionTerm,
                  FaceVariable, ImplicitSourceTerm, TransientTerm, VanLeerConvectionTerm, serialComm)
from fipy.solvers import LinearLUSolver

from streamer_drift_diffusion._transport_kernel import MAX_FACE_RATIO, face_fluxes
from streamer_drift_diffusion.transport import ElectronTransport, _centered_stencil


def fipy_mesh(grid):
    """共有格子と同じ寸法・番号付けのFiPy円筒メッシュを作る。"""
    nr, nz = grid.shape
    return CylindricalGrid2D(nr=nr, nz=nz, dr=grid.spacing, dz=grid.spacing, communicator=serialComm)


def fipy_poisson_matrix(poisson):
    """PoissonSolverと同じ左辺を、FiPyのDiffusionTerm・ImplicitSourceTermで組む。"""
    grid = poisson.grid
    mesh = fipy_mesh(grid)
    low, high = grid.adjacent(poisson.gas)
    potential = CellVariable(mesh=mesh, value=0.)
    equation = (-DiffusionTerm(coeff=FaceVariable(mesh=mesh, value=(low & high).astype(float)))
                + ImplicitSourceTerm(coeff=CellVariable(mesh=mesh, value=poisson.diagonal.ravel(order='F')))
                == CellVariable(mesh=mesh, value=0.))
    equation.cacheMatrix()
    equation.justResidualVector(var=potential, solver=LinearLUSolver())
    return equation.matrix.matrix.tocsc()


class _VanLeer(VanLeerConvectionTerm):
    """FiPyのVan Leer補正に、境界近傍の気体側勾配を組み込む。"""

    def __init__(self, velocity, grid, mesh, gas):
        super().__init__(coeff=velocity)
        valid = _centered_stencil(gas)
        faces = np.flatnonzero(mesh.interiorFaces)
        axis = np.argmax(grid.directions[:, faces], axis=0)
        cells = np.asarray(mesh.faceCellIDs[:, faces])
        one_sided = grid.interior(gas)[faces] & ~valid[axis, cells]
        self._boundary_indices = np.flatnonzero(one_sided.any(axis=0))
        self._boundary_faces = faces[self._boundary_indices]
        self._one_sided = one_sided[:, self._boundary_indices]
        axis = axis[self._boundary_indices]
        cells = cells[:, self._boundary_indices]
        # 軸対称な二次分布で、最初のセル中心の勾配は隣接差分の半分。
        on_axis = (axis == 0) & (cells % grid.shape[0] == 0)
        self._slope_scale = np.where(on_axis, .5, 1.) / grid.spacing
        self._area = grid.area[self._boundary_faces]
        self._volume = grid.volume.ravel(order="F")[cells]

    def _getWeight(self, var, transientGeomCoeff=None, diffusionGeomCoeff=None):
        # 風上側は速度の符号だけで決まる。軸などのゼロ面で0/0を避ける。
        return super()._getWeight(var, transientGeomCoeff=transientGeomCoeff,
                                  diffusionGeomCoeff=None)

    def _getOldAdjacentValues(self, oldArray, id1, id2, dt):
        values = np.asarray(super()._getOldAdjacentValues(oldArray, id1, id2, dt))
        old = np.asarray(oldArray.value)
        center = np.stack((old[id1], old[id2]))
        boundary = self._boundary_indices
        boundary_center = center[:, boundary]
        face_rate = np.abs(np.asarray(self._getGeomCoeff(oldArray))[self._boundary_faces])
        # 欠けた側の差分を気体側の差分で補い、FiPyと同じ時間補正を使う。
        slope = (boundary_center[::-1] - boundary_center) * self._slope_scale
        one_sided = boundary_center + .5*slope*(self._volume - dt*face_rate)/self._area
        values[:, boundary] = np.where(self._one_sided, one_sided, values[:, boundary])
        # 流出面値の上界をCFL評価と共有する。
        return tuple(np.clip(values, 0., MAX_FACE_RATIO*center))


class FiPyElectronTransport(ElectronTransport):
    """ElectronTransportと同じAPIで、NumPy係数とFiPy方程式により1回更新する参照実装。"""

    def __init__(self, state, spacing):
        super().__init__(state, spacing)
        self.mesh = fipy_mesh(self.grid)
        self.fipy_density = CellVariable(mesh=self.mesh, value=0., hasOld=True)
        self.velocity = FaceVariable(mesh=self.mesh, rank=1, value=0.)
        self.diffusion = FaceVariable(mesh=self.mesh, value=0.)
        self.absorption = CellVariable(mesh=self.mesh, value=0.)
        self.equation = (TransientTerm()
                         + _VanLeer(self.velocity, self.grid, self.mesh, self.gas)
                         - ExplicitDiffusionTerm(coeff=self.diffusion)
                         + self.absorption * self.fipy_density.old)
        self.solver = LinearLUSolver(tolerance=1e-12)

    def prepare(self, state):
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
        low, high = grid.adjacent(coefficient)
        diffusion = low + high
        np.divide(2*low*high, low+high, out=diffusion, where=self.interior & (diffusion > 0.))
        self.diffusion_rates = diffusion*self._coefficient_area/self.distance
        self.boundary_rates = (self._coefficient_area*np.maximum(np.where(self.low_gas, speed, -speed), 0.)
                               + self.diffusion_rates)*self.boundary
        diffusion_loss = grid.sum_faces(self.diffusion_rates)
        # 内部面の再構築密度は最大で旧セル密度の2倍。吸収面は旧セル値を使う。
        self.advection_rates = speed*self._coefficient_area
        rate = self.advection_rates*np.where(self.interior, MAX_FACE_RATIO, 1.)
        loss = grid.sum_faces(np.maximum(rate, 0.), np.maximum(-rate, 0.)) + diffusion_loss
        maxima = (float(np.max((values/grid.volume)[self.gas])) for values in (loss, diffusion_loss))
        self.transport_limit, self.diffusion_limit = (1/value if value > 0. else math.inf for value in maxima)
        self.velocity.setValue(grid.directions*speed*self.interior)
        self.diffusion.setValue(diffusion*self.interior)
        self.absorption.setValue((grid.sum_faces(self.boundary_rates)/grid.volume).ravel(order='F'))
        return self

    def step(self, state, dt, *, prepared=False):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("transport dt must be finite and positive")
        if not prepared:
            self.prepare(state)
        if not np.all(np.isfinite(state.ne)):
            raise FloatingPointError("nonfinite transport density")
        grid = self.grid
        density = self.fipy_density
        density.setValue(state.ne.ravel(order='F'))
        density.updateOld()
        old = state.ne
        low_density, high_density = grid.adjacent(old)
        losses = 2*np.pi*dt*np.bincount(grid.boundary_kind,
            weights=self.boundary_rates*np.where(self.low_gas, low_density, high_density), minlength=3)
        self.equation.solve(var=density, dt=dt, solver=self.solver)
        value = np.array(density.value)
        faces = self._copy_faces
        if faces.size:
            flux = face_fluxes(np.asarray(density.old.value), self._centered,
                *grid.face_ids, grid.low, grid.high, self.low_gas, self.interior,
                grid.area, self._volume, self.advection_rates, self.diffusion_rates,
                self.boundary_rates, self._boundary_sources, grid.spacing, dt)
            outward = np.where(self.low_gas[faces], flux[faces], -flux[faces])
            cells = np.where(self.low_gas[faces], grid.low[faces], grid.high[faces])
            value -= dt*np.bincount(cells, weights=outward, minlength=self._volume.size)/self._volume
            losses += 2*np.pi*dt*np.bincount(grid.boundary_kind[faces], weights=outward, minlength=3)
        if not np.all(np.isfinite(value)) or not np.all(np.isfinite(losses)):
            raise FloatingPointError("nonfinite FiPy transport solution or boundary loss")
        state.ne[:] = value.reshape(self.shape, order='F')
        return losses
