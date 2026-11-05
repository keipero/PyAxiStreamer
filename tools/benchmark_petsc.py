"""PETScの求解法と、本体の固定行列・LU再利用を比較する計測専用ツール。"""
from contextlib import contextmanager, nullcontext
from pathlib import Path
import argparse
import json
import os
import platform
import sys
import tempfile
import time
from unittest.mock import patch


@contextmanager
def rebuild_poisson():
    """計測中だけPoissonの毎回組立・LU分解を選択する。

    Yields
    ------
    None
        with文の範囲で初期化する状態はreuse_matrix=Falseとなる。

    Notes
    -----
    本体のPoissonSolverを使い、比較のためコンストラクタ引数だけを変更する。
    独立した求解処理は持たない。終了時に通常の初期化へ戻す。

    Examples
    --------
    再組立との比較から呼ぶ::

        with rebuild_poisson():
            state = run_simulation(config, write_output=False, verbose=False)
    """
    from functools import partial
    from streamer_drift_diffusion.electrostatics import PoissonSolver

    with patch('streamer_drift_diffusion.initial_conditions.PoissonSolver',
               partial(PoissonSolver, reuse_matrix=False)):
        yield


def compare_solvers(config, repeats, *, rhs_tolerance=None):
    """同じPoisson問題で、組立を含むLU・反復法の時間と精度を比較する。

    Parameters
    ----------
    config : SimulationConfig
        格子・電極形状・残差許容値を指定する設定。
    repeats : int
        ウォームアップ1回を除いた計測回数。
    rhs_tolerance : float, optional
        反復法の相対2ノルム許容値。Noneなら独立基準/(10*sqrt(ncells))。
        指定しても本体の最大相対残差の検査基準は変えない。

    Returns
    -------
    list of dict
        各手法の時間 [s]、独立残差、LU解からの差、収束情報。失敗も記録する。

    Notes
    -----
    単一の初期化済み状態を使用し、各求解は本体と同じく初期電位0から開始する。
    既定の反復法の2ノルム許容値はmax相対残差基準の1/(10*sqrt(ncells))倍とする。
    各求解で電荷源を少し変える。順序はLU、GMRES/ILU、CG/Jacobi、CG/ICC、CG/GAMG、GMRES/GAMG。

    Examples
    --------
    小格子を調べる::

        results = compare_solvers(SimulationConfig(nr=20, nz=15), repeats=3)
    """
    import numpy as np
    from fipy.solvers.petsc import LinearGMRESSolver, LinearCGSolver
    from streamer_drift_diffusion import initialize_simulation

    with rebuild_poisson():
        state = initialize_simulation(config)
    operator = state.poisson
    original = operator.solver
    r, z = state.rh[:, None], state.zh[None, :]
    charge = 1e13*np.exp(-(r/2e-4)**2-((z-7e-4)/1e-4)**2)
    results, reference = [], None
    for method in ('lu', 'gmres_ilu', 'cg_jacobi', 'cg_icc', 'cg_gamg', 'gmres_gamg'):
        samples, residuals = [], []
        try:
            if method == 'lu':
                operator.solver = original
            else:
                family, pc = method.split('_')
                factory = LinearGMRESSolver if family == 'gmres' else LinearCGSolver
                tolerance = config.poisson_tolerance/(10*np.sqrt(config.nr*config.nz)) if rhs_tolerance is None else rhs_tolerance
                operator.solver = factory(tolerance=tolerance,
                    absolute_tolerance=0., criterion='RHS', iterations=4000, precon=pc)
            for repeat in range(repeats+1):
                start = time.perf_counter()
                answer = operator.solve(charge*(1+repeat*.001), config.voltage)
                elapsed = time.perf_counter()-start
                if repeat:
                    samples.append(elapsed)
                residuals.append(operator.residual)
            if method == 'lu':
                reference = answer.copy()
            error = float(np.max(np.abs(answer-reference))/max(np.max(np.abs(reference)), 1e-300))
            results.append(dict(method=method, samples_s=samples, median_s=float(np.median(samples)),
                max_residual=max(residuals), lu_scale_error=error,
                solver_relative_tolerance=float(operator.solver.tolerance),
                symmetric=bool(operator.equation.matrix.matrix.isSymmetric(tol=1e-13)),
                convergence=str(operator.solver.convergence)))
        except Exception as exc:
            results.append(dict(method=method, error=f'{type(exc).__name__}: {exc}'))
        print('Poisson:', results[-1], flush=True)
    return results


def compare_runs(config, repeats):
    """毎回組立LUと本体の固定行列再利用を、出力あり・なしで交互に計測する。

    Parameters
    ----------
    config : SimulationConfig
        両手法に共通の格子・物理条件・終了時刻。
    repeats : int
        各手法・各I/O条件の計測回数。各1回のウォームアップは除外する。

    Returns
    -------
    list of dict
        初期化・組立・求解・時間発展・任意のI/Oを含む時間 [s]と最終状態の差。

    Raises
    ------
    AssertionError
        ne・phi・中心電場・累積境界流出数のスケール相対誤差が1e-8を超える場合。

    Notes
    -----
    各試行は新しい状態を作り、一時フォルダへ出力する。Python起動とimportは含めない。
    各対の物理量を検査し、計測順序は対ごとに反転する。並行計測は行わない。

    Examples
    --------
    10ステップの計算を調べる::

        results = compare_runs(SimulationConfig(t_end=1e-11), repeats=3)
    """
    import numpy as np
    from streamer_drift_diffusion import run_simulation

    results = []
    for output in (False, True):
        samples = {'rebuild': [], 'reuse': []}
        errors = {}
        for repeat in range(repeats+1):
            snapshots = {}
            order = list(samples) if repeat % 2 == 0 else list(reversed(samples))
            for method in order:
                with tempfile.TemporaryDirectory() as directory:
                    with rebuild_poisson() if method == 'rebuild' else nullcontext():
                        start = time.perf_counter()
                        state = run_simulation(config, write_output=output, output_dir=directory, verbose=False)
                        elapsed = time.perf_counter()-start
                        if repeat:
                            samples[method].append(elapsed)
                        snapshots[method] = {name: getattr(state, name).copy() for name in
                            ('ne', 'phi', 'ex', 'ey', 'abs_e', 'electron_losses')}
                print(f'Run: output={output} repeat={repeat} method={method} seconds={elapsed:.6f}', flush=True)
            for name, values in snapshots['rebuild'].items():
                error = float(np.max(np.abs(snapshots['reuse'][name]-values))/max(np.max(np.abs(values)), 1e-300))
                assert np.isfinite(error) and error <= 1e-8, (name, error)
                errors[name] = max(error, errors.get(name, 0.))
        results.append(dict(write_output=output, samples_s=samples,
            median_s={key: float(np.median(value)) for key, value in samples.items()}, max_scale_errors=errors))
    return results


def profile_run(config):
    """出力なしの本体実行をcProfileで調べる。

    Parameters
    ----------
    config : SimulationConfig
        計測する格子と終了時刻。求解法は本体の再利用LU。

    Returns
    -------
    dict
        プロファイラ付きの経過時間 [s]と、累積時間順の上位25項目。

    Notes
    -----
    Python起動・importは含めない。プロファイラの負荷があるため、通常の速度比較とは分けて読む。

    Examples
    --------
    計算の内訳だけを調べる::

        result = profile_run(SimulationConfig(t_end=1e-11))
    """
    import cProfile
    import io
    import pstats
    from streamer_drift_diffusion import run_simulation

    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.runcall(run_simulation, config, write_output=False, verbose=False)
    elapsed = time.perf_counter()-start
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats('cumulative').print_stats(25)
    return dict(seconds=elapsed, cumulative_top25=stream.getvalue())


def main():
    """1プロセス・1スレッドのPETSc性能検討を実行し、任意でJSONへ保存する。

    Returns
    -------
    int
        完了時は0。連成計算や精度検査の失敗は例外で停止する。

    Notes
    -----
    FiPyのimport前にPETScを選び、OpenMP/OpenBLAS/MKLのスレッド数を1にする。
    PETSC_OPTIONSが空の新しいプロセスで実行する。独立したベンチマークを同時実行しない。
    Poisson単体の手法選択と、計算全体の再利用効果を分けて評価する。

    Examples
    --------
    プロジェクトフォルダで、PETScを利用できるPythonから呼ぶ::

        python tools/benchmark_petsc.py --report PETSC_PERFORMANCE_RESULTS.json
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--profile', action='store_true', help='Add a separate cProfile run')
    parser.add_argument('--rhs-tolerance', type=float, help='Override iterative relative 2-norm tolerance')
    args = parser.parse_args()
    if args.repeats < 1 or args.steps < 1 or os.environ.get('PETSC_OPTIONS', '').strip():
        parser.error('repeats and steps must be positive; PETSC_OPTIONS must be empty')
    if args.rhs_tolerance is not None and not 0 < args.rhs_tolerance < 1:
        parser.error('rhs-tolerance must be finite and between 0 and 1')
    os.environ['FIPY_SOLVERS'] = 'petsc'
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    sys.stdout.reconfigure(encoding='utf-8')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np
    import scipy
    import fipy
    from petsc4py import PETSc
    from mpi4py import MPI
    from dataclasses import asdict
    from streamer_drift_diffusion import SimulationConfig

    if MPI.COMM_WORLD.size != 1 or np.dtype(PETSc.ScalarType) != np.dtype(np.float64):
        raise RuntimeError('Requires one MPI process and real float64 PETSc')
    config = SimulationConfig(t_end=args.steps*1e-12)
    report = dict(environment=dict(python=sys.version, processor=platform.processor(), numpy=np.__version__,
        scipy=scipy.__version__, fipy=fipy.__version__, petsc=PETSc.Sys.getVersion(),
        mpi=MPI.Get_library_version().strip(), threads=1), config=asdict(config), repeats=args.repeats)
    report['poisson'] = compare_solvers(config, args.repeats, rhs_tolerance=args.rhs_tolerance)
    report['coupled'] = compare_runs(config, args.repeats)
    if args.profile:
        report['profile'] = profile_run(config)
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
