"""PETScを使用しないSciPy LU再利用・Numba共有メモリ並列の計測用試作。"""
from contextlib import contextmanager, ExitStack
import argparse
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch


def make_transport_kernel():
    """面流束とセルごとの発散を計算するNumba並列カーネルを生成する。

    Returns
    -------
    numba.core.registry.CPUDispatcher
        各面・セルを別々の反復へ割り当て、共有配列への競合書込みを避けるカーネル。

    Notes
    -----
    一次上流・陽的拡散・半セル吸収の面係数を既存grid/prepare()から受け取る。
    fastmathは使用しない。反応・物理係数は既存のNumPy処理を維持する。

    Examples
    --------
    計測開始時に生成する::

        kernel = make_transport_kernel()
    """
    import numpy as np
    from numba import njit, prange

    @njit(parallel=True, cache=True, fastmath=False)
    def update(old, volume, gas, low, high, cell_faces, speed, area, diffusion, boundary, interior, low_gas, dt):
        """同じ旧値から面流束を作り、セルごとに新しい密度を返す。

        Parameters
        ----------
        old, volume, gas : ndarray
            FiPyセル順の旧密度 [m⁻³]、体積 [m³、2π省略]と気体マスク。
        low, high, cell_faces : ndarray of int
            各面の隣接セルIDと、各セルの4面ID。領域外のセルIDは-1。
        speed, area : ndarray
            面の座標正方向の速度 [m/s]と面積 [m²、2π省略]。
        diffusion, boundary : ndarray
            既存prepare()の拡散・吸収係数 [m³/s]。
        interior, low_gas : ndarray of bool
            気体間面と、低座標側が気体かどうか。
        dt : float
            更新幅 [s]。

        Returns
        -------
        ndarray
            新しい密度 [m⁻³]。入力配列は変更しない。

        Notes
        -----
        面に一度だけ流束を書き、セルごとに4面を読み取る2段階で競合を避ける。
        金属セルへ吸収した電子は蓄積しない。クリップやfastmathは使用しない。
        """
        flux = np.zeros(low.size)
        result = np.empty_like(old)
        for face in prange(low.size):
            if interior[face]:
                left, right = old[low[face]], old[high[face]]
                upwind = left if speed[face] >= 0. else right
                flux[face] = area[face]*speed[face]*upwind+diffusion[face]*(left-right)
            elif boundary[face] != 0.:
                if low_gas[face]:
                    flux[face] = boundary[face]*old[low[face]]
                else:
                    flux[face] = -boundary[face]*old[high[face]]
        for cell in prange(old.size):
            net = 0.
            if gas[cell]:
                for index in range(4):
                    face = cell_faces[index, cell]
                    net += flux[face] if low[face] == cell else -flux[face]
            result[cell] = old[cell]-dt*net/volume[cell]
        return result
    return update


@contextmanager
def prototype(reuse, kernel=None):
    """この計測中だけSciPy LU再利用と、任意でNumba輸送を適用する。

    Parameters
    ----------
    reuse : bool
        FiPyで作ったSciPy行列のLU分解を再利用するかどうか。
    kernel : CPUDispatcher, optional
        NumPy旧密度から新密度を計算するNumbaカーネル。NoneならFiPy輸送を維持する。

    Yields
    ------
    None
        with文の範囲だけ実行関数を差し替える。

    Notes
    -----
    本体への採用前の試作。初回のFiPy求解と追加LU分解も時間に含める。
    PETScやMPIは使用しない。試作のLUや状態はwith終了時に解放する。

    Examples
    --------
    SciPy環境の計測から呼ぶ::

        with prototype(True, kernel):
            state = run_simulation(config, write_output=False, verbose=False)
    """
    import numpy as np
    from scipy.sparse.linalg import splu
    from streamer_drift_diffusion.electrostatics import PoissonSolver
    from streamer_drift_diffusion.transport import ElectronTransport
    original = PoissonSolver.solve
    systems = {}

    def solve(operator, rho, voltage):
        """固定したFiPy/SciPy行列のLUを再利用して電位を計算する。

        Parameters
        ----------
        operator : PoissonSolver
            既存FiPy方程式と正本電位。
        rho : ndarray
            電荷密度/epsilon0 [V/m²]。
        voltage : float
            印加電圧 [V]。

        Returns
        -------
        ndarray
            正本と共有する2次元電位 [V]。

        Notes
        -----
        最大相対残差は本体と同じ指標で検査する。精度評価用の内部試作。
        """
        if operator not in systems:
            original(operator, rho, voltage)
            matrix = operator.equation.matrix.matrix.copy()
            systems[operator] = (matrix, splu(matrix.tocsc()))
        matrix, factor = systems[operator]
        source = np.where(operator.gas, rho, 0.)+voltage*operator.boundary_source
        rhs = (source*operator.grid.volume).ravel(order='F')
        values = factor.solve(rhs)
        operator.residual = float(np.max(np.abs(matrix@values-rhs))/max(np.max(np.abs(rhs)), np.finfo(float).tiny))
        if not np.all(np.isfinite(values)) or not np.isfinite(operator.residual) or operator.residual > operator.tolerance:
            raise RuntimeError(f'SciPy prototype residual: {operator.residual}')
        operator.potential.setValue(values)
        operator.potential.setValue(voltage, where=~operator.gas.ravel(order='F'))
        return operator.potential.value.reshape(operator.grid.shape, order='F')

    def step(operator, state, dt, *, prepared=False):
        """既存の面係数でNumba輸送を行い、元と同じ境界流出数を返す。

        Parameters
        ----------
        operator : ElectronTransport
            既存のgrid・面係数・密度正本を持つ輸送オブジェクト。
        state : SimulationState
            現在の計算状態。
        dt : float
            有限で正の更新幅 [s]。
        prepared : bool, optional
            既存prepare()の係数が準備済みかどうか。

        Returns
        -------
        ndarray, shape (3,)
            電極・接地・外端の流出数。2πを含む。

        Notes
        -----
        既存CFL・吸収係数と旧値を使う。入力配列を並列に上書きせず、新値の有限性検査後に反映する。
        """
        if not np.isfinite(dt) or dt <= 0 or not np.all(np.isfinite(state.ne)):
            raise FloatingPointError('Invalid shared-memory transport input')
        if not prepared:
            operator.prepare(state)
        operator.density.updateOld()
        grid = operator.grid
        old = operator.density.old.value
        values = kernel(old, grid.volume.ravel(order='F'), operator.gas.ravel(order='F'), grid.low, grid.high,
            np.asarray(grid.mesh.cellFaceIDs), grid.pack_faces(state.vex, state.vey), grid.area,
            operator.diffusion_rates, operator.boundary_rates, operator.interior, operator.low_gas, dt)
        low, high = grid.adjacent(old.reshape(grid.shape, order='F'))
        losses = 2*np.pi*dt*np.bincount(grid.boundary_kind,
            weights=operator.boundary_rates*np.where(operator.low_gas, low, high), minlength=3)
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(losses)):
            raise FloatingPointError('Nonfinite shared-memory transport output')
        operator.density.setValue(values)
        return losses

    with ExitStack() as stack:
        if reuse:
            stack.enter_context(patch.object(PoissonSolver, 'solve', solve))
        if kernel is not None:
            stack.enter_context(patch.object(ElectronTransport, 'step', step))
        yield


def main():
    """FiPy/SciPyと、LU再利用・Numbaの1/2/4/8スレッドを比較する。

    Returns
    -------
    int
        成功時は0。精度比較・計算の失敗は例外で停止する。

    Notes
    -----
    PETSc・mpi4pyのないCPU用環境を推奨する。FiPyのバックエンドはscipyに固定する。
    JITの初回はウォームアップとして除外し、その時間も別に保存する。
    計測は初期化・全時間発展を含み、出力なし。反応は既存のNumPy処理。

    Examples
    --------
    CPU用環境から実行する::

        python tools/benchmark_shared_memory.py --report SHARED_MEMORY_RESULTS.json
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nr', type=int, default=200)
    parser.add_argument('--nz', type=int, default=150)
    parser.add_argument('--delta', type=float, default=1e-5)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['FIPY_SOLVERS'] = 'scipy'
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np
    import scipy
    import numba
    import fipy
    from streamer_drift_diffusion import SimulationConfig, run_simulation
    assert fipy.solvers.solver_suite == 'scipy'
    kernel = make_transport_kernel()
    config = SimulationConfig(nr=args.nr, nz=args.nz, delta=args.delta, t_end=args.steps*1e-12)
    report = dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, numba=numba.__version__,
        shape=[args.nr,args.nz], delta=args.delta, steps=args.steps, repeats=args.repeats, results=[])
    reference = None
    modes = [('fipy_scipy', False, 0), ('scipy_reuse', True, 0)]+[(f'numba_{count}', True, count) for count in (1,2,4,8)]
    for name, reuse, threads in modes:
        samples, errors = [], {}
        for repeat in range(args.repeats+1):
            if threads:
                numba.set_num_threads(threads)
            with prototype(reuse, kernel if threads else None):
                start = time.perf_counter()
                state = run_simulation(config, write_output=False, verbose=False)
                elapsed = time.perf_counter()-start
            if repeat:
                samples.append(elapsed)
            else:
                warmup = elapsed
            snapshot = {key: getattr(state,key).copy() for key in ('ne','phi','ex','ey','abs_e','electron_losses')}
            if reference is None:
                reference = snapshot
            for key, values in snapshot.items():
                scale = max(np.max(np.abs(reference[key])), 1e-300)
                error = float(np.max(np.abs(values-reference[key]))/scale)
                assert np.isfinite(error) and error <= 1e-8, (name,key,error)
                errors[key] = max(error, errors.get(key,0.))
            print(f'{name}: repeat={repeat} seconds={elapsed:.6f}',flush=True)
        report['results'].append(dict(mode=name, numba_threads=threads, warmup_s=warmup, samples_s=samples,
            median_s=float(np.median(samples)), max_scale_errors=errors))
        args.report.write_text(json.dumps(report, indent=2)+'\n',encoding='utf-8')
    report['threading_layer'] = numba.threading_layer()
    report['petsc_loaded'] = sys.modules.get('petsc4py') is not None
    report['mpi_loaded'] = sys.modules.get('mpi4py') is not None
    args.report.write_text(json.dumps(report, indent=2)+'\n',encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
