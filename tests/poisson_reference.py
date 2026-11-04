"""独立した小格子の円筒FV直接解。FiPyの面番号・組立コードは使わない。"""
import numpy as np


def dense_poisson(state, voltage, source=None):
    nr, nz = state.ne.shape
    h = state.grid.spacing
    matrix = np.zeros((nr*nz, nr*nz))
    rhs = np.zeros(nr*nz)
    source = np.zeros((nr, nz)) if source is None else source
    for i, j in np.ndindex((nr, nz)):
        row = i*nz+j
        if not state.gas_mask[i, j]:
            matrix[row, row], rhs[row] = 1/h**2, voltage/h**2
            continue
        rhs[row] = source[i, j]
        for ni, nj, weight in ((i-1, j, i/(i+.5)/h**2),
                                (i+1, j, (i+1)/(i+.5)/h**2),
                                (i, j-1, 1/h**2), (i, j+1, 1/h**2)):
            if weight == 0.:
                continue
            if 0 <= ni < nr and 0 <= nj < nz and state.gas_mask[ni, nj]:
                matrix[row, ni*nz+nj] -= weight
                matrix[row, row] += weight
            else:
                boundary = (voltage*state.radial_boundary[j] if ni == nr else
                            voltage*state.upper_boundary[i] if nj == nz else
                            0. if nj < 0 else voltage)
                matrix[row, row] += 2*weight
                rhs[row] += 2*weight*boundary
    return matrix, rhs
