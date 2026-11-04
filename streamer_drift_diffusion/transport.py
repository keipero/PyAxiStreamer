"""FiPyによる円筒有限体積の陽的電子移流拡散と、吸収境界の粒子収支。"""
import numpy as np

from fipy import (CellVariable, CylindricalGrid2D, ExplicitDiffusionTerm,
                  ExplicitUpwindConvectionTerm, FaceVariable, TransientTerm,
                  serialComm)
from fipy.solvers import LinearLUSolver


def face_areas(sr, sz):
    """2πを省略した全半径面・全軸方向面の面積を返す。"""
    return (np.concatenate((sr, 2*sr[-1:]-sr[-2:-1]), axis=0),
            np.concatenate((sz, sz[:, -1:]), axis=1))


def transport_rates(vr, vz, sr, sz):
    """全セル面の速度×面積 [m³/s]。軸面の輸送は常に0。"""
    ar, az = face_areas(sr, sz)
    if vr.shape != ar.shape or vz.shape != az.shape:
        raise ValueError("velocities must have radial (nr+1,nz) and axial (nr,nz+1) face shapes")
    radial, axial = vr * ar, vz * az
    radial[0] = 0.
    return radial, axial


def diffusion_rates(diffusion, gas_mask, spacing, sr, sz):
    """D×面積/距離 [m³/s]。気体間は調和平均、吸収面は半セル距離。"""
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("diffusion requires a finite positive spacing [m]")
    coefficient = np.where(gas_mask, diffusion, 0.)
    if np.any(~np.isfinite(coefficient)) or np.any(coefficient < 0):
        raise ValueError("diffusion coefficients must be finite and nonnegative")
    rates = []
    for axis, area in enumerate(face_areas(sr, sz)):
        low_pad, high_pad = [(0, 0), (0, 0)], [(0, 0), (0, 0)]
        low_pad[axis], high_pad[axis] = (1, 0), (0, 1)
        low, high = np.pad(coefficient, low_pad), np.pad(coefficient, high_pad)
        both = np.pad(gas_mask, low_pad) & np.pad(gas_mask, high_pad)
        face = low + high
        np.divide(2*low*high, low+high, out=face, where=both & (face > 0.))
        rates.append(face * area / np.where(both, spacing, spacing/2))
    return rates[0], rates[1]


class _ExplicitUpwind(ExplicitUpwindConvectionTerm):
    """一次上流の重みをDと独立に求め、FiPy 4.0.xのゼロ面の0/0を避ける。

    正のTransientTermに対する上流側は速度の符号だけで決まる。
    標準実装に拡散幾何係数を渡さないことで、D=0かつ流速0の面や軸で
    不要なPeclet数の除算を行わない。物理係数や数値例外の設定は変えない。
    """
    def _getWeight(self, var, transientGeomCoeff=None, diffusionGeomCoeff=None):
        return super()._getWeight(var, transientGeomCoeff=transientGeomCoeff,
                                  diffusionGeomCoeff=None)


class ElectronTransport:
    """格子・方程式を再利用する電子輸送。1シミュレーションにつき1個作る。

    stateの中心配列は(nr,nz)、速度は全セル面、deは中心のD [m²/s]。
    電極形状と等間隔格子は初期化後に変更しない。
    step()はneだけを更新し、流出数を返す。累積と試行の受理は呼出側が行う。
    FiPyの現在値・旧値は各試行の開始時にstateから同期する。
    """
    def __init__(self, state, spacing):
        if not np.isfinite(spacing) or spacing <= 0:
            raise ValueError("transport requires a finite positive spacing [m]")
        self.shape = state.ne.shape
        self.spacing = spacing
        nr, nz = self.shape
        self.gas = state.gas_mask.copy()
        self.volume = state.volume
        self.ar, self.az = face_areas(state.sr, state.sz)
        self.mesh = CylindricalGrid2D(nr=nr, nz=nz, dr=spacing, dz=spacing,
                                      communicator=serialComm)
        if not np.allclose(self.mesh.cellVolumes, self.volume.ravel(order='F'),
                           rtol=1e-13, atol=0.):
            raise ValueError("transport volumes must match the cylindrical mesh")
        self.both_r = np.zeros((nr+1, nz), dtype=bool)
        self.both_z = np.zeros((nr, nz+1), dtype=bool)
        self.both_r[1:-1] = self.gas[:-1] & self.gas[1:]
        self.both_z[:, 1:-1] = self.gas[:, :-1] & self.gas[:, 1:]

        # 面番号の実装に依存せず、座標と面法線から対応を一度作る。
        coords = np.asarray(self.mesh.faceCenters) / spacing
        self.radial = np.abs(np.asarray(self.mesh.faceNormals)[0]) > .5
        self.axial = ~self.radial
        self.ri = np.rint(coords[0, self.radial]).astype(int)
        self.rj = np.rint(coords[1, self.radial]-.5).astype(int)
        self.zi = np.rint(coords[0, self.axial]-.5).astype(int)
        self.zj = np.rint(coords[1, self.axial]).astype(int)

        # 各方向で隣が気体でない面だけを吸収項へ移す。
        self.boundary_masks = (
            self.gas & ~np.pad(self.gas[:-1], ((1, 0), (0, 0))),
            self.gas & ~np.pad(self.gas[1:], ((0, 1), (0, 0))),
            self.gas & ~np.pad(self.gas[:, :-1], ((0, 0), (1, 0))),
            self.gas & ~np.pad(self.gas[:, 1:], ((0, 0), (0, 1))),
        )
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
        """旧密度から陽的に更新し、[電極, 接地, 外側]の流出電子数を返す。

        返却値だけが2πを含む。負密度のクリップは行わない。
        解・流出量が非有限なら、state.neへの反映前にFloatingPointErrorを送出する。
        """
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("transport dt must be finite and positive")
        if state.ne.shape != self.shape or not np.array_equal(state.gas_mask, self.gas):
            raise ValueError("transport grid and electrode mask must not change")
        if (not np.all(np.isfinite(state.ne)) or not np.all(np.isfinite(state.vex))
                or not np.all(np.isfinite(state.vey))):
            raise FloatingPointError("nonfinite transport density or velocity")

        radial, axial = transport_rates(state.vex, state.vey, state.sr, state.sz)
        dr, dz = diffusion_rates(state.de, self.gas, self.spacing, state.sr, state.sz)
        # 吸収率Kは体積で割る前の[m³/s]。流出数にも同じKを使う。
        sides = ((np.maximum(-radial[:-1], 0.) + dr[:-1]) * self.boundary_masks[0],
                 (np.maximum(radial[1:], 0.) + dr[1:]) * self.boundary_masks[1],
                 (np.maximum(-axial[:, :-1], 0.) + dz[:, :-1]) * self.boundary_masks[2],
                 (np.maximum(axial[:, 1:], 0.) + dz[:, 1:]) * self.boundary_masks[3])
        losses_by_cell = np.zeros((3, *self.shape))
        # 外端分を別区分へ移す。軸は面積0なので寄与しない。
        losses_by_cell[2, -1] += sides[1][-1]
        sides[1][-1] = 0.
        losses_by_cell[1, :, 0] += sides[2][:, 0]
        sides[2][:, 0] = 0.
        losses_by_cell[2, :, -1] += sides[3][:, -1]
        sides[3][:, -1] = 0.
        losses_by_cell[0] = sum(sides)

        # 気体間だけをFiPyの面項へ渡す。境界は上の損失項が担当する。
        face_d = np.zeros(self.mesh.numberOfFaces)
        diffusion_r, diffusion_z = np.zeros_like(dr), np.zeros_like(dz)
        np.divide(dr*self.spacing, self.ar, out=diffusion_r, where=self.both_r)
        np.divide(dz*self.spacing, self.az, out=diffusion_z, where=self.both_z)
        face_d[self.radial] = diffusion_r[self.ri, self.rj]
        face_d[self.axial] = diffusion_z[self.zi, self.zj]
        velocity = np.zeros((2, self.mesh.numberOfFaces))
        velocity[0, self.radial] = (state.vex*self.both_r)[self.ri, self.rj]
        velocity[1, self.axial] = (state.vey*self.both_z)[self.zi, self.zj]

        old = np.where(self.gas, state.ne, 0.)
        self.density.setValue(old.ravel(order='F'))
        self.density.updateOld()
        self.velocity.setValue(velocity)
        self.diffusion.setValue(face_d)
        self.absorption.setValue((losses_by_cell.sum(axis=0)/self.volume).ravel(order='F'))
        self.equation.solve(var=self.density, dt=dt, solver=self.solver)
        result = np.asarray(self.density.value).reshape(self.shape, order='F')
        losses = 2*np.pi*dt*np.sum(losses_by_cell*old, axis=(1, 2))
        if not np.all(np.isfinite(result)) or not np.all(np.isfinite(losses)):
            raise FloatingPointError("nonfinite FiPy transport solution or boundary loss")
        state.ne[:] = result
        return losses
