"""MPI領域分割案の基礎検証。円筒格子・所有範囲・halo・FiPy方程式を調べる。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def inspect_domain(nr, nz):
    """FiPyのz方向分割と通信・方程式を小格子で検証する。

    Parameters
    ----------
    nr, nz : int
        全体の半径・軸方向セル数。セル幅1の円筒格子を使用する。

    Returns
    -------
    dict or None
        rank 0に所有範囲・誤差・全体解を返し、ほかのrankはNoneを返す。

    Raises
    ------
    AssertionError
        所有範囲の重複・欠落、座標・halo・既知解・保存則の検査に失敗した場合。

    Notes
    -----
    本体のMPI実装ではない。FiPy 4.0.3のprivateな所有IDと_updateGhosts()を検証する。
    空rankがある構成では分割情報だけを検証し、方程式求解はスキップする。
    最終配列のgatherはこの小格子検証に限定する。

    Examples
    --------
    MPIから起動したプロセス内で呼ぶ::

        result = inspect_domain(5, 9)
    """
    import numpy as np
    import fipy
    from fipy import CylindricalGrid2D, CellVariable, FaceVariable, DiffusionTerm, ExplicitDiffusionTerm, TransientTerm, parallelComm
    from fipy.solvers.petsc import LinearCGSolver, LinearLUSolver, JacobiPreconditioner
    from mpi4py import MPI
    from petsc4py import PETSc
    from streamer_drift_diffusion.transport import _ExplicitUpwind

    comm = MPI.COMM_WORLD
    mesh = CylindricalGrid2D(nr=nr, nz=nz, dr=1., dz=1., overlap=2, communicator=parallelComm)
    gids = np.asarray(mesh._globalOverlappingCellIDs)
    owned = np.asarray(mesh._localNonOverlappingCellIDs)
    owned_gids = np.asarray(mesh._globalNonOverlappingCellIDs)
    owned_faces = np.asarray(mesh._localNonOverlappingFaceIDs)
    centers = np.asarray(mesh.cellCenters)
    np.testing.assert_allclose(centers[0], gids % nr+.5)
    np.testing.assert_allclose(centers[1], gids // nr+.5)
    faces = np.asarray(mesh.faceCenters)
    physical = np.isclose(faces[0], 0.) | np.isclose(faces[0], nr) | np.isclose(faces[1], 0.) | np.isclose(faces[1], nz)
    artificial = np.asarray(mesh.exteriorFaces) & ~physical
    layout = dict(rank=comm.rank, cells=int(mesh.numberOfCells), owned=int(owned.size),
        owned_z_columns=sorted(set((owned_gids//nr).tolist())), local_z_columns=sorted(set((gids//nr).tolist())),
        owned_faces=int(owned_faces.size), artificial_exterior_faces=int(artificial.sum()),
        owned_artificial_exterior_faces=int(artificial[owned_faces].sum()))
    layouts = comm.gather(layout, root=0)
    ownership = comm.gather(owned_gids.tolist(), root=0)
    total_owned = comm.allreduce(int(owned.size), op=MPI.SUM)
    assert total_owned == nr*nz
    total_faces = comm.allreduce(int(owned_faces.size), op=MPI.SUM)
    global_face_ids = np.asarray(mesh._globalNonOverlappingFaceIDs)
    face_ownership = comm.gather(global_face_ids.tolist(), root=0)
    if comm.rank == 0:
        flattened = [item for group in face_ownership for item in group]
        print('Face layout:',dict(ranks=comm.size, total_ids=total_faces, unique_ids=len(set(flattened)),
            expected=nr*(nz+1)+(nr+1)*nz, layouts=layouts),flush=True)
        assert sorted(set(flattened)) == list(range(nr*(nz+1)+(nr+1)*nz))
    assert comm.allreduce(int(artificial[owned_faces].sum()), op=MPI.SUM) == 0
    empty = comm.allreduce(int(owned.size == 0), op=MPI.SUM)
    result = None
    if comm.rank == 0:
        assert sorted(item for group in ownership for item in group) == list(range(nr*nz))
        result = dict(nr=nr, nz=nz, ranks=comm.size, overlap=2, fipy=fipy.__version__,
            petsc=PETSc.Sys.getVersion(), layout=layouts, empty_ranks=empty,
            nonoverlap_face_id_count=total_faces, unique_face_count=len(set(flattened)))
    if empty:
        if result is not None:
            result['numerical_probe'] = 'skipped: empty ranks; initial implementation should reject this layout'
        return result

    mask = np.zeros(mesh.numberOfCells, dtype=bool)
    mask[owned] = True
    halo = CellVariable(mesh=mesh, value=np.where(mask, 11.+3*gids, -999.))
    halo._updateGhosts()
    halo_error = comm.allreduce(float(np.max(np.abs(halo.value-(11.+3*gids)))), op=MPI.MAX)
    assert halo_error == 0.

    potential = CellVariable(mesh=mesh, value=0.)
    potential.constrain(0., np.isclose(faces[1], 0.))
    potential.constrain(float(nz), np.isclose(faces[1], nz))
    poisson = DiffusionTerm(coeff=1.)
    poisson.cacheMatrix()
    poisson.cacheRHSvector()
    poisson.solve(var=potential, solver=LinearCGSolver(tolerance=1e-13, absolute_tolerance=0.,
        criterion='RHS', iterations=2000, precon='gamg'))
    potential_error = comm.allreduce(float(np.max(np.abs(potential.value-centers[1]))), op=MPI.MAX)
    assert potential_error < 1e-10
    residual = poisson.matrix*potential.value-poisson.RHSvector
    denominator = comm.allreduce(float(np.max(np.abs(poisson.RHSvector[owned]))), op=MPI.MAX)
    relative = comm.allreduce(float(np.max(np.abs(residual[owned]))), op=MPI.MAX)/denominator
    assert relative <= 1e-10

    density = CellVariable(mesh=mesh, hasOld=True, value=1.+np.exp(-((centers[1]-nz/2)/2.)**2))
    velocity = np.zeros((2, mesh.numberOfFaces))
    velocity[1, ~physical] = 1.
    equation = TransientTerm() == ExplicitDiffusionTerm(coeff=.05)-_ExplicitUpwind(coeff=FaceVariable(mesh=mesh, rank=1, value=velocity))
    solver = LinearLUSolver(tolerance=1e-12, criterion='initial')
    solver.preconditioner = JacobiPreconditioner()
    volume = np.asarray(mesh.cellVolumes)
    initial = comm.allreduce(float(np.sum(density.value[owned]*volume[owned])), op=MPI.SUM)
    for _ in range(3):
        density._updateGhosts()
        density.updateOld()
        equation.solve(var=density, dt=.1, solver=solver)
    final = comm.allreduce(float(np.sum(density.value[owned]*volume[owned])), op=MPI.SUM)
    assert abs(final-initial)/initial < 1e-12
    gathered = comm.gather((owned_gids.tolist(), potential.value[owned].tolist(), density.value[owned].tolist()), root=0)
    if result is not None:
        potential_global, density_global = np.empty(nr*nz), np.empty(nr*nz)
        for ids, phi, ne in gathered:
            potential_global[ids], density_global[ids] = phi, ne
        result.update(halo_error=halo_error, laplace_error=potential_error, poisson_residual=relative,
            transport_mass_error=abs(final-initial)/initial, potential=potential_global.tolist(), density=density_global.tolist())
    return result


def main():
    """WindowsのローカルMPIで1・2・4rankを順に検証し、JSONへ保存する。

    Returns
    -------
    int
        成功時は0。子プロセスの失敗・逐次解との不一致は例外で停止する。

    Notes
    -----
    通常のシミュレーションは変更しない。5×9格子で数値検証し、5×3の4rankで空rankも調べる。
    PETSc環境のConda経由で実行する。計測用で各プロセスのOpenMP/BLASを1、I_MPI_PINを0にする。

    Examples
    --------
    プロジェクトフォルダから実行する::

        python tools/probe_mpi_domain.py --report MPI_DOMAIN_PROBE_RESULTS.json
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--nr', type=int, default=5)
    parser.add_argument('--nz', type=int, default=9)
    args = parser.parse_args()
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['FIPY_SOLVERS'], os.environ['I_MPI_PIN'] = 'petsc', '0'
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    if args.worker:
        result = inspect_domain(args.nr, args.nz)
        if result is not None:
            args.report.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
        return 0
    import numpy as np
    launcher = Path(sys.prefix)/'Library/bin/mpiexec.exe'
    results = []
    with tempfile.TemporaryDirectory() as directory:
        for ranks, nz in ((1, 9), (2, 9), (4, 9), (4, 3)):
            path = Path(directory)/f'{ranks}_{nz}.json'
            command = [str(launcher), '-localonly', '-localroot', '-n', str(ranks), sys.executable, '-E', str(Path(__file__).resolve()),
                '--worker', '--nr', '5', '--nz', str(nz), '--report', str(path)]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=90,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            if completed.returncode:
                raise RuntimeError((completed.stderr+completed.stdout)[-6000:])
            result = json.loads(path.read_text(encoding='utf-8'))
            if nz == 9 and ranks != 1:
                for name in ('potential', 'density'):
                    reference = np.asarray(results[0][name])
                    values = np.asarray(result[name])
                    np.testing.assert_allclose(values, reference, rtol=1e-11, atol=1e-12)
                    result[name+'_serial_scale_error'] = float(np.max(np.abs(values-reference))/np.max(np.abs(reference)))
            results.append(result)
            args.report.write_text(json.dumps(results, indent=2)+'\n', encoding='utf-8')
            print(dict(ranks=ranks, nz=nz, empty_ranks=result['empty_ranks'],
                halo_error=result.get('halo_error'), poisson_residual=result.get('poisson_residual'),
                transport_error=result.get('density_serial_scale_error', 0.)), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
