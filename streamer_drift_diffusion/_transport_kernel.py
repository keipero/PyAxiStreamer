"""共有の円筒有限体積格子に対する陽的な電子輸送。"""

import numpy as np
from numba import njit


MAX_FACE_RATIO = 2.


@njit(cache=True)
def prepare_coefficients(speed, coefficient, gas, radial, axial, low, high,
                         low_gas, high_gas, interior, area, volume, distance):
    """面係数を作り、セルごとの最大流出頻度を集計する。"""
    advection = np.empty_like(speed)
    diffusion = np.empty_like(speed)
    absorption = np.zeros_like(speed)
    for face in range(speed.size):
        left_gas, right_gas = low_gas[face], high_gas[face]
        left = coefficient[low[face]] if left_gas else 0.
        right = coefficient[high[face]] if right_gas else 0.
        value = left + right
        if interior[face] and value > 0.:
            value = 2.*left*right/value
        diffusion[face] = value*area[face]/distance[face]
        advection[face] = speed[face]*area[face]
        if left_gas != right_gas:
            outward = speed[face] if left_gas else -speed[face]
            absorption[face] = area[face]*max(outward, 0.) + diffusion[face]
        if not (np.isfinite(advection[face]) and np.isfinite(diffusion[face])
                and np.isfinite(absorption[face])):
            raise FloatingPointError("nonfinite transport face coefficients")

    transport_max, diffusion_max = 0., 0.
    nr = radial.shape[0] - 1
    for j in range(axial.shape[1] - 1):
        for i in range(nr):
            cell = i + nr*j
            if not gas[cell]:
                continue
            faces = (radial[i, j], radial[i+1, j], axial[i, j], axial[i, j+1])
            diff, drift = 0., 0.
            for side, face in enumerate(faces):
                diff += diffusion[face]
                outward = advection[face] if side % 2 else -advection[face]
                drift += max(outward, 0.)*(MAX_FACE_RATIO if interior[face] else 1.)
            diffusion_max = max(diffusion_max, diff/volume[cell])
            transport_max = max(transport_max, (diff + drift)/volume[cell])
    if not np.isfinite(transport_max):
        raise FloatingPointError("nonfinite transport frequency")
    return advection, diffusion, absorption, transport_max, diffusion_max


@njit(cache=True)
def transport_update(old, gas, centered, radial, axial, low, high, low_gas,
                     interior, area, volume, advection, diffusion, absorption,
                     boundary_kind, spacing, dt):
    """旧密度から新密度と、吸収境界を通過した電子数を返す。"""
    nr, nz = radial.shape[0] - 1, axial.shape[1] - 1
    # FiPyと同じ、面の算術平均と面積から求めるGauss勾配。
    gradient = np.zeros((2, old.size))
    for j in range(nz):
        for i in range(nr):
            cell = i + nr*j
            density = old[cell]
            if centered[0, cell]:
                lower = .5*(density + old[cell-1])*area[radial[i, j]]
                upper = .5*(density + old[cell+1])*area[radial[i+1, j]]
                gradient[0, cell] = (upper - lower)/volume[cell]
            if centered[1, cell]:
                lower = .5*(density + old[cell-nr])*area[axial[i, j]]
                upper = .5*(density + old[cell+nr])*area[axial[i, j+1]]
                gradient[1, cell] = (upper - lower)/volume[cell]

    flux = np.zeros(area.size)
    losses = np.zeros(3)
    for face in range(area.size):
        if interior[face]:
            left, right = low[face], high[face]
            difference = old[right] - old[left]
            flow = advection[face]
            flux[face] = -diffusion[face]*difference
            if flow != 0.:
                donor = left if flow > 0. else right
                direction = 1. if flow > 0. else -1.
                # Fortran順の隣接差は半径方向で1、軸方向でnr。
                axis = 0 if right - left == 1 else 1
                downwind = direction*difference/spacing
                slope = downwind
                if centered[axis, donor]:
                    upwind = 2.*direction*gradient[axis, donor] - downwind
                    slope = 0.
                    if (upwind > 0. and downwind > 0.) or (upwind < 0. and downwind < 0.):
                        magnitude = min(2.*abs(upwind), 2.*abs(downwind),
                                        .5*(abs(upwind) + abs(downwind)))
                        slope = magnitude if upwind > 0. else -magnitude
                elif axis == 0 and donor % nr == 0:
                    slope *= .5
                center = old[donor]
                density = center + .5*slope*(volume[donor] - dt*abs(flow))/area[face]
                density = min(max(density, 0.), MAX_FACE_RATIO*center)
                flux[face] += flow*density
        elif absorption[face] != 0.:
            donor = low[face] if low_gas[face] else high[face]
            rate = absorption[face]*old[donor]
            flux[face] = rate if low_gas[face] else -rate
            losses[boundary_kind[face]] += rate

    new = old.copy()
    for j in range(nz):
        for i in range(nr):
            cell = i + nr*j
            if gas[cell]:
                net = (flux[radial[i, j]] - flux[radial[i+1, j]]
                       + flux[axial[i, j]] - flux[axial[i, j+1]])
                new[cell] += dt*net/volume[cell]
    return new, (2.*np.pi*dt)*losses
