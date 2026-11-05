"""SciPy Poissonの初回・反復求解と、出力なしの時間発展を計測する。"""
import argparse
import json
import os
from pathlib import Path
import sys
import time


def measure(config, repeats, updates):
    """初回分解と右辺更新、初期化を含む連成計算を別々に測定する。

    Parameters
    ----------
    config : SimulationConfig
        格子・物理・時間発展の条件。
    repeats : int
        ウォームアップを除く反復計測数。
    updates : int
        Poisson単体の1サンプル中の右辺更新数。

    Returns
    -------
    dict
        初回求解秒数、Poisson反復・連成計算の各サンプルと中央値、残差。

    Notes
    -----
    Poisson単体の初回は行列組立・分解・求解・残差検査を含む。
    反復は同じソルバーへ電荷源を0.1%ずつ変更する。
    連成計算は毎回状態を新しく作り、全計算を測る。CSV出力とPython起動・importは除く。
    各反復の最初の1サンプルはウォームアップとして別に記録する。

    Examples
    --------
    SciPyバックエンドの環境から呼ぶ::

        result = measure(SimulationConfig(nr=8, nz=12, t_end=1e-12), 3, 20)
    """
    import numpy as np
    from streamer_drift_diffusion import initialize_simulation, run_simulation
    state = initialize_simulation(config)
    operator = state.poisson
    charge = np.random.default_rng(72).normal(size=state.ne.shape)*state.gas_mask*1e12
    start = time.perf_counter()
    operator.solve(charge, config.voltage)
    first = time.perf_counter()-start
    poisson, coupled = [], []
    residual = operator.residual
    try:
        for repeat in range(repeats+1):
            start = time.perf_counter()
            for update in range(updates):
                operator.solve(charge*(1.+.001*update), config.voltage)
                residual = max(residual, operator.residual)
            elapsed = time.perf_counter()-start
            if repeat:
                poisson.append(elapsed)
            else:
                poisson_warmup = elapsed
        for repeat in range(repeats+1):
            start = time.perf_counter()
            run_simulation(config, write_output=False, verbose=False)
            elapsed = time.perf_counter()-start
            if repeat:
                coupled.append(elapsed)
            else:
                coupled_warmup = elapsed
            print(f'coupled repeat={repeat}: {elapsed:.6f} s', flush=True)
    finally:
        if hasattr(operator, 'close'):
            operator.close()
    return dict(poisson_first_s=first, poisson_updates=updates, poisson_samples_s=poisson,
        poisson_median_s=float(np.median(poisson)), poisson_warmup_s=poisson_warmup,
        poisson_max_residual=residual, coupled_samples_s=coupled,
        coupled_median_s=float(np.median(coupled)), coupled_warmup_s=coupled_warmup)


def main():
    """同じスレッド設定のSciPy計算を計測し、JSONへ保存する。

    Returns
    -------
    int
        計測成功時は0。不正な引数・バックエンド・計算失敗は例外で停止する。

    Notes
    -----
    別checkoutとの比較には--project-rootへそのルートを指定できる。
    物理条件・環境・反復回数を揃え、ほかの計測と同時に実行しない。
    このツールは計算本体の求解方式を差し替えない。

    Examples
    --------
    プロジェクトルートのSciPy環境から実行する::

        python tools/benchmark_scipy_lu.py --steps 20 --report SCIPY_LU_BENCHMARK.json
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--nr', type=int, default=200)
    parser.add_argument('--nz', type=int, default=150)
    parser.add_argument('--delta', type=float, default=1e-5)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--updates', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if min(args.steps, args.updates, args.repeats) < 1:
        parser.error('steps, updates and repeats must be positive')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['FIPY_SOLVERS'] = 'scipy'
    os.environ['I_MPI_PIN'] = '0'
    sys.path.insert(0, str(args.project_root.resolve()))
    import numpy as np
    import scipy
    import fipy
    import streamer_drift_diffusion
    from streamer_drift_diffusion import SimulationConfig
    assert fipy.solvers.solver_suite == 'scipy'
    config = SimulationConfig(nr=args.nr, nz=args.nz, delta=args.delta, t_end=args.steps*1e-12)
    report = dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, fipy=fipy.__version__,
        fipy_solver_suite=fipy.solvers.solver_suite, module_path=streamer_drift_diffusion.__file__,
        shape=[args.nr, args.nz], delta=args.delta, steps=args.steps, repeats=args.repeats,
        threads={key: os.environ[key] for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
    report['measurements'] = measure(config, args.repeats, args.updates)
    args.report.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report['measurements'], indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
