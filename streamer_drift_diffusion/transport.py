"""FiPyによる円筒有限体積の陽的電子移流拡散と、吸収境界の粒子収支。"""
import numpy as np

from fipy import (CellVariable, ExplicitDiffusionTerm, ExplicitUpwindConvectionTerm,
                  FaceVariable, TransientTerm)
from fipy.solvers import LinearLUSolver



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
