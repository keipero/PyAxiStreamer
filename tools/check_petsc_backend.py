"""PETSc・FiPy・ストリーマ計算を1プロセスで検証する実行用チェック。"""
import os
from pathlib import Path
import sys


def main():
    """実行環境と、小さい既知解・連成計算によるPETSc接続を確認する。

    Returns
    -------
    int
        全検査が成功した場合は0。環境情報と各検査の結果を標準出力へ表示する。

    Raises
    ------
    ImportError
        PETSc・MPI・FiPyまたは必要なDLLを読み込めない場合。
    RuntimeError
        PETSc以外のバックエンド、複数MPIプロセス、または倍精度実数以外の構成。
    AssertionError
        既知解との差、連成計算の有限性、またはPoisson残差の検査に失敗した場合。

    Notes
    -----
    FiPyをimportする前にFIPY_SOLVERS=petscを設定する。新しいPythonプロセスで呼ぶ。
    2×2のPETSc直接解、FiPyの1次元拡散既知解、小格子のストリーマ連成を順に検証する。
    計算結果ファイルは作成しない。MPIランタイムが利用できる環境で実行する。

    Examples
    --------
    プロジェクトフォルダから、PETSc環境のPythonで実行する::

        python tools/check_petsc_backend.py
    """
    sys.stdout.reconfigure(encoding='utf-8')
    os.environ['FIPY_SOLVERS'] = 'petsc'
    import numpy as np
    from mpi4py import MPI
    from petsc4py import PETSc
    import petsc4py

    print('Python:', sys.executable, flush=True)
    print('NumPy:', np.__version__, np.__file__, flush=True)
    print('petsc4py:', petsc4py.__version__, 'PETSc:', PETSc.Sys.getVersion(), flush=True)
    print('MPI:', MPI.Get_library_version().strip(), flush=True)
    if MPI.COMM_WORLD.size != 1 or np.dtype(PETSc.ScalarType) != np.dtype(np.float64):
        raise RuntimeError('This check requires one MPI process and real float64 PETSc')

    matrix = PETSc.Mat().createAIJ(size=(2, 2), nnz=2, comm=PETSc.COMM_SELF)
    matrix.setValues([0, 1], [0, 1], [[4., 1.], [1., 3.]])
    matrix.assemble()
    solution, rhs = matrix.createVecRight(), matrix.createVecLeft()
    solver = PETSc.KSP().create(comm=PETSc.COMM_SELF)
    try:
        rhs.setValues([0, 1], [1., 2.])
        rhs.assemble()
        solver.setOperators(matrix)
        solver.setType('preonly')
        solver.getPC().setType('lu')
        solver.setErrorIfNotConverged(True)
        solver.solve(rhs, solution)
        np.testing.assert_allclose(solution.getArray(readonly=True), [1./11., 7./11.], rtol=1e-13)
        print('PETSc standalone LU: PASS', flush=True)
    finally:
        for obj in (solver, solution, rhs, matrix):
            obj.destroy()

    import fipy
    from fipy import Grid1D, CellVariable, DiffusionTerm, serialComm
    from fipy.solvers import LinearLUSolver
    print('FiPy:', fipy.__version__, 'Suite:', fipy.solvers.solver_suite, flush=True)
    if fipy.solvers.solver_suite != 'petsc':
        raise RuntimeError('Run this check in a fresh process with the PETSc backend')
    mesh = Grid1D(nx=8, dx=1./8., communicator=serialComm)
    value = CellVariable(mesh=mesh, value=0.)
    value.constrain(0., mesh.facesLeft)
    value.constrain(1., mesh.facesRight)
    DiffusionTerm(coeff=1.).solve(var=value, solver=LinearLUSolver(tolerance=1e-12, criterion='RHS'))
    np.testing.assert_allclose(value.value, np.asarray(mesh.cellCenters[0]), rtol=1e-12)
    print('FiPy diffusion LU: PASS', flush=True)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from streamer_drift_diffusion import SimulationConfig, run_simulation
    state = run_simulation(SimulationConfig(nr=4, nz=6, delta=1e-4, electrode_gap=4e-4,
        cathode_radius=2e-4, z0=2e-4, voltage=50., nmax=1e10, background_ionization=0.,
        electron_diffusion=.5, t_end=2e-12), write_output=False, verbose=False)
    assert isinstance(state.poisson.solver, LinearLUSolver) and state.poisson.reuse_matrix
    assert not hasattr(state.poisson.solver, 'ksp')
    assert np.all(np.isfinite(state.ne)) and np.all(np.isfinite(state.phi))
    assert state.poisson.residual <= 1e-10
    print('Streamer coupled solve: PASS; residual =', state.poisson.residual, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
