"""再利用後の独立条件CPU並列と、Poisson線形系だけのMPI分散を評価する。"""
import argparse
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


@contextmanager
def binary_viewer(path, mode, comm):
    """PETScバイナリViewerを開き、正常・異常終了とも解放する。

    Parameters
    ----------
    path : str or Path
        ファイルの保存先・読込み元。
    mode : str
        読込みrまたは書込みw。
    comm : PETSc.Comm
        ファイル操作に参加するコミュニケータ。

    Yields
    ------
    PETSc.Viewer
        使用後にdestroy()するViewer。

    Examples
    --------
    MPIの行列読込みから使用する::

        with binary_viewer(path, 'r', PETSc.COMM_WORLD) as viewer:
            matrix = PETSc.Mat().load(viewer)
    """
    from petsc4py import PETSc
    viewer = PETSc.Viewer().createBinary(str(path), mode, comm=comm)
    try:
        yield viewer
    finally:
        viewer.destroy()


def run_case(payload):
    """1プロセスで独立した条件を計算し、結果のダイジェストを返す。

    Parameters
    ----------
    payload : tuple of (float, int)
        電圧 [V]と固定ステップ数。標準200×150格子、dt=1 ps、出力なし。

    Returns
    -------
    dict
        電圧、ne・phi・中心電場・累積流出数のSHA256。状態・PETScオブジェクトは転送しない。

    Examples
    --------
    PETSc環境から呼ぶ::

        result = run_case((4000., 10))
    """
    import numpy as np
    from streamer_drift_diffusion import SimulationConfig, run_simulation
    voltage, steps = payload
    state = run_simulation(SimulationConfig(voltage=voltage, t_end=steps*1e-12), write_output=False, verbose=False)
    return dict(voltage=voltage, hashes={name: hashlib.sha256(np.ascontiguousarray(getattr(state, name)).tobytes()).hexdigest()
        for name in ('ne', 'phi', 'ex', 'ey', 'abs_e', 'electron_losses')})


def benchmark_batches(steps, repeats, workers):
    """独立4条件を逐次実行とWindows spawnのプロセスプールで比較する。

    Parameters
    ----------
    steps, repeats : int
        各条件の固定ステップ数と、初回を除いた測定回数。
    workers : list of int
        1は同一プロセスの逐次実行、2以上はプールのプロセス数。先頭は1とする。

    Returns
    -------
    list of dict
        初回の起動・importを含む時間、プール維持時の時間 [s]、逐次解との完全一致結果。

    Notes
    -----
    電圧は-4000、-2000、2000、4000 V。各条件は全格子の独立計算であり、領域分割ではない。
    各プールは全反復の間保持する。各バッチの順序を揃え、全結果のハッシュを検査する。

    Examples
    --------
    計測から呼ぶ::

        results = benchmark_batches(10, 3, [1, 2, 4])
    """
    import statistics
    from contextlib import nullcontext
    cases = [(voltage, steps) for voltage in (-4000., -2000., 2000., 4000.)]
    results, reference = [], None
    for count in workers:
        samples = []
        start = time.perf_counter()
        pool = ProcessPoolExecutor(max_workers=count, mp_context=multiprocessing.get_context('spawn')) if count > 1 else nullcontext(None)
        with pool as executor:
            for repeat in range(repeats+1):
                if repeat:
                    start = time.perf_counter()
                values = list(executor.map(run_case, cases)) if executor else list(map(run_case, cases))
                elapsed = time.perf_counter()-start
                if reference is None:
                    reference = values
                if values != reference:
                    raise AssertionError(f'Parallel results differ for workers={count}')
                if repeat:
                    samples.append(elapsed)
                else:
                    cold = elapsed
                print(f'Batch: workers={count} repeat={repeat} seconds={elapsed:.6f}', flush=True)
        results.append(dict(workers=count, cases=4, cold_batch_s=cold, samples_s=samples,
            median_s=statistics.median(samples), exact=True))
    return results


def mpi_worker(args):
    """FiPyで保存したPoisson行列をMPI分散読込みし、固定行列求解を測定する。

    Parameters
    ----------
    args : argparse.Namespace
        行列・右辺・LU参照解、手法、ステップ数・測定回数と保存先。

    Returns
    -------
    None
        rank 0がJSONを保存する。誤差・残差異常は例外で停止する。

    Notes
    -----
    Poisson線形系だけの評価。物理配列・輸送・反応・出力の領域分割を実装したものではない。
    右辺全体を各ステップ0.1%ずつ変え、独立した最大相対残差1e-10を毎回検査する。
    行列読込みと前処理の構築は更新時間から分けて記録する。

    Examples
    --------
    親のbenchmark_mpi()がmpiexecから起動する。
    """
    import numpy as np
    from mpi4py import MPI
    from petsc4py import PETSc
    import statistics
    comm = MPI.COMM_WORLD
    comm.Barrier()
    start = time.perf_counter()
    with binary_viewer(args.matrix, 'r', PETSc.COMM_WORLD) as viewer:
        matrix = PETSc.Mat().load(viewer)
    with binary_viewer(args.rhs, 'r', PETSc.COMM_WORLD) as viewer:
        original = PETSc.Vec().load(viewer)
    rhs, solution, residual = (original.duplicate() for _ in range(3))
    ksp = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
    try:
        load_s = comm.allreduce(time.perf_counter()-start, op=MPI.MAX)
        ksp.setOperators(matrix)
        ksp.setType('preonly' if args.method == 'lu' else 'cg')
        ksp.getPC().setType(args.method)
        ksp.setTolerances(rtol=1e-10/(10*matrix.getSize()[0]**.5), atol=0., max_it=4000)
        ksp.setErrorIfNotConverged(True)
        comm.Barrier()
        start = time.perf_counter()
        ksp.setUp()
        setup_s = comm.allreduce(time.perf_counter()-start, op=MPI.MAX)
        samples, max_residual = [], 0.
        for repeat in range(args.repeats+1):
            comm.Barrier()
            start = time.perf_counter()
            for step in range(args.steps):
                original.copy(rhs)
                rhs.scale(1+.001*step)
                solution.set(0.)
                ksp.solve(rhs, solution)
                matrix.mult(solution, residual)
                residual.axpy(-1., rhs)
                relative = residual.norm(PETSc.NormType.NORM_INFINITY)/max(rhs.norm(PETSc.NormType.NORM_INFINITY), np.finfo(float).tiny)
                if not np.isfinite(relative) or relative > 1e-10 or ksp.reason <= 0:
                    raise RuntimeError(f'MPI Poisson residual: {relative}')
                max_residual = max(max_residual, relative)
            elapsed = comm.allreduce(time.perf_counter()-start, op=MPI.MAX)
            if repeat:
                samples.append(elapsed)
        chunks = comm.gather(solution.getArray(readonly=True).copy(), root=0)
        if comm.rank == 0:
            expected = np.load(args.reference)*(1+.001*(args.steps-1))
            error = float(np.max(np.abs(np.concatenate(chunks)-expected))/np.max(np.abs(expected)))
            if not np.isfinite(error) or error > 1e-8:
                raise AssertionError(f'MPI solution error: {error}')
            result = dict(ranks=comm.size, method=args.method, load_s=load_s, setup_s=setup_s,
                samples_s=samples, median_s=statistics.median(samples), max_residual=max_residual,
                lu_scale_error=error, last_iterations=ksp.its)
            args.report.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    finally:
        for obj in (ksp, residual, solution, rhs, original, matrix):
            obj.destroy()


def benchmark_mpi(steps, repeats):
    """同じFiPy行列で1ランクLUと1・2・4ランクGAMGを比較する。

    Parameters
    ----------
    steps, repeats : int
        各測定での右辺更新数と測定回数。

    Returns
    -------
    list of dict
        線形系だけの並列時間 [s]と精度。起動失敗はエラー内容を記録する。

    Notes
    -----
    導入済みConda環境のLibrary/bin/mpiexec.exeを使用し、全ジョブはこのPCだけで実行する。
    通常のシミュレーションをmpiexecで領域分割する機能は追加しない。

    Examples
    --------
    計測から呼ぶ::

        results = benchmark_mpi(10, 3)
    """
    import numpy as np
    from petsc4py import PETSc
    from streamer_drift_diffusion import SimulationConfig, initialize_simulation
    launcher = Path(sys.prefix)/'Library/bin/mpiexec.exe'
    if not launcher.is_file():
        return [dict(error=f'MPI launcher not found: {launcher}')]
    state = initialize_simulation(SimulationConfig())
    r, z = state.rh[:, None], state.zh[None, :]
    reference = state.poisson.solve(1e13*np.exp(-(r/2e-4)**2-((z-7e-4)/1e-4)**2), 4000.).copy()
    results = []
    try:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            matrix_path, rhs_path, reference_path = (folder/name for name in ('matrix.bin', 'rhs.bin', 'reference.npy'))
            with binary_viewer(matrix_path, 'w', PETSc.COMM_SELF) as viewer:
                state.poisson.equation.matrix.matrix.view(viewer)
            rhs = PETSc.Vec().createWithArray(state.poisson.equation.RHSvector.copy(), comm=PETSc.COMM_SELF)
            try:
                with binary_viewer(rhs_path, 'w', PETSc.COMM_SELF) as viewer:
                    rhs.view(viewer)
            finally:
                rhs.destroy()
            np.save(reference_path, reference.ravel(order='F'))
            for ranks, method in ((1, 'lu'), (1, 'gamg'), (2, 'gamg'), (4, 'gamg')):
                report = folder/f'{ranks}_{method}.json'
                command = [str(launcher), '-localonly', '-localroot', '-n', str(ranks), sys.executable, '-E', str(Path(__file__).resolve()),
                    '--mpi-worker', '--matrix', str(matrix_path), '--rhs', str(rhs_path), '--reference', str(reference_path),
                    '--method', method, '--steps', str(steps), '--repeats', str(repeats), '--report', str(report)]
                start = time.perf_counter()
                completed = subprocess.run(command, capture_output=True, text=True, timeout=120,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                result = json.loads(report.read_text(encoding='utf-8')) if completed.returncode == 0 else dict(
                    ranks=ranks, method=method, error=(completed.stderr+completed.stdout)[-3000:])
                result['process_wall_s'] = time.perf_counter()-start
                results.append(result)
                print('MPI:', result, flush=True)
    finally:
        state.poisson.close()
    return results


def main():
    """CPU並列評価とMPI求解評価を行い、JSONへ保存する。

    Returns
    -------
    int
        完了時は0。独立条件の不一致や計算失敗は例外で停止する。

    Notes
    -----
    新しいプロセスで呼ぶ。各プロセスのOpenMP/BLASスレッドは1に指定する。
    独立プロセス同士のコア固定の競合を避けるため、この計測ではI_MPI_PIN=0とする。
    GPU対応PETScは前提にせず、このツールではCPUだけを測定する。

    Examples
    --------
    プロジェクトフォルダで、Conda経由のPythonから実行する::

        python tools/benchmark_parallel.py --report PETSC_PARALLEL_RESULTS.json
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--workers', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--mpi-only', action='store_true')
    parser.add_argument('--mpi-worker', action='store_true')
    parser.add_argument('--matrix', type=Path)
    parser.add_argument('--rhs', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--method', choices=['lu', 'gamg'], default='gamg')
    args = parser.parse_args()
    if args.steps < 1 or args.repeats < 1 or args.workers[0] != 1 or any(value < 1 for value in args.workers):
        parser.error('positive steps/repeats/workers required; workers must start with 1')
    if os.environ.get('PETSC_OPTIONS', '').strip():
        parser.error('Run with empty PETSC_OPTIONS')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['FIPY_SOLVERS'] = 'petsc'
    os.environ['I_MPI_PIN'] = '0'
    sys.stdout.reconfigure(encoding='utf-8')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    if args.mpi_worker:
        mpi_worker(args)
        return 0
    report = dict(steps=args.steps, repeats=args.repeats, logical_cpus=os.cpu_count(), threads_per_process=1,
        mpi_pinning=False)
    if not args.mpi_only:
        report['independent_cases'] = benchmark_batches(args.steps, args.repeats, args.workers)
        args.report.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    report['mpi_poisson_only'] = benchmark_mpi(args.steps, args.repeats)
    args.report.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
