"""SciPy/FiPyと共有メモリ試作を、拡散・波形・刻み制御の12条件で比較する。"""
import argparse
import csv
from dataclasses import fields
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest


def snapshot(state, folder):
    """状態配列とCSVの各列を独立した比較対象として取り出す。

    Parameters
    ----------
    state : SimulationState
        計算終了時の状態。配列はコピーして保持する。
    folder : pathlib.Path
        保存したフレームと刻み履歴のフォルダ。

    Returns
    -------
    dict
        状態配列、CSVの数値列または文字列列。メタデータJSONは比較しない。

    Examples
    --------
    計算直後に使用する::

        values = snapshot(state, output_dir)
    """
    import numpy as np
    result = {item.name: getattr(state, item.name).copy() for item in fields(state)
              if isinstance(getattr(state, item.name), np.ndarray)}
    result.update(ne=state.ne.copy(), phi=state.phi.copy())
    for path in sorted(folder.glob('*.csv')):
        with path.open(encoding='utf-8', newline='') as stream:
            rows = list(csv.reader(stream))
        try:
            result['csv/'+path.name] = np.asarray(rows, dtype=float)
        except ValueError:
            for index, name in enumerate(rows[0]):
                column = [row[index] for row in rows[1:]]
                try:
                    result[f'csv/{path.name}/{name}'] = np.asarray(column, dtype=float)
                except ValueError:
                    result[f'csv/{path.name}/{name}'] = column
    return result


def compare(reference, candidate):
    """配列と文字列を比較し、状態・CSV別の最大スケール誤差を返す。

    Parameters
    ----------
    reference, candidate : dict
        snapshot()で取得した同じ条件の結果。

    Returns
    -------
    dict
        最大スケール誤差とビット単位一致の有無。CSVの丸め誤差は別集計。

    Raises
    ------
    AssertionError
        形状、非有限値の位置、文字列が不一致、または許容誤差を超える。

    Notes
    -----
    有限値の誤差はmax|差|/max(max|基準|, 1e-300)。状態は1e-8以内、
    小数点以下6桁のCSVは2e-6以内を要求する。NaNは認めない。
    """
    import numpy as np
    assert reference.keys() == candidate.keys()
    errors = dict(state=0., csv=0., bitwise_equal=True)
    for name, expected in reference.items():
        actual = candidate[name]
        if isinstance(expected, list):
            assert expected == actual, name
            continue
        assert expected.shape == actual.shape, name
        if expected.dtype.kind in 'biu':
            assert np.array_equal(expected, actual), name
            continue
        assert not np.any(np.isnan(actual)), name
        assert np.array_equal(np.isfinite(expected), np.isfinite(actual)), name
        finite = np.isfinite(expected)
        assert np.array_equal(expected[~finite], actual[~finite]), name
        scale = max(float(np.max(np.abs(expected[finite]), initial=0.)), 1e-300)
        error = float(np.max(np.abs(actual[finite]-expected[finite]), initial=0.)/scale)
        kind = 'csv' if name.startswith('csv/') else 'state'
        assert error <= (2e-6 if kind == 'csv' else 1e-8), (name, error)
        errors[kind] = max(errors[kind], error)
        errors['bitwise_equal'] &= bool(np.array_equal(expected, actual))
    return errors


def main():
    """12条件で既存FiPy、LU再利用、Numbaの1/4スレッドを検証する。

    Returns
    -------
    int
        全比較成功時は0。不一致は例外で停止する。

    Notes
    -----
    PETSc/MPIのないCPU環境で実行する。状態・全保存CSVを比較する。
    試作の本体採用に必要な異常系・全回帰テストを代替するものではない。

    Examples
    --------
    CPU環境から実行する::

        python tools/validate_shared_memory.py --report SHARED_MEMORY_NUMERICAL_RESULTS.json
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    os.environ['FIPY_SOLVERS'] = 'scipy'
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numba
    from streamer_drift_diffusion import SimulationConfig, run_simulation
    from tools.benchmark_shared_memory import make_transport_kernel, prototype
    kernel = make_transport_kernel()
    report = dict(python=sys.version, cases=[], state_tolerance=1e-8, csv_tolerance=2e-6)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        waveform = root/'waveform.txt'
        waveform.write_text('0 0\n0.02 1\n0.04 -0.5\n0.11 0\n', encoding='utf-8')
        for index, (diffusion, adaptive, voltage) in enumerate(itertools.product((0., .5), (False, True), ('positive', 'negative', 'impulse'))):
            config = SimulationConfig(nr=8, nz=12, delta=1e-4, electrode_gap=6e-4,
                cathode_radius=2e-4, nmax=1e16, background_ionization=1e11, z0=5e-4,
                dt=1e-11, t_end=1.1e-10, output_interval=2e-11, electron_diffusion=diffusion,
                adaptive_timestep=adaptive, voltage=-500. if voltage == 'negative' else 500.,
                voltage_mode='impulse' if voltage == 'impulse' else 'dc', impulse_peak_voltage=500.,
                voltage_waveform_file=str(waveform))
            case = dict(diffusion=diffusion, adaptive=adaptive, voltage=voltage, comparisons={})
            reference = serial = None
            for mode, reuse, threads in (('fipy_scipy', False, 0), ('scipy_reuse', True, 0), ('numba_1', True, 1), ('numba_4', True, 4)):
                if threads:
                    numba.set_num_threads(threads)
                folder = root/f'{index}_{mode}'
                with prototype(reuse, kernel if threads else None):
                    state = run_simulation(config, output_dir=folder, verbose=False)
                values = snapshot(state, folder)
                if reference is None:
                    reference = values
                    case['comparison_arrays'] = len(values)
                else:
                    case['comparisons'][mode] = compare(reference, values)
                if threads == 1:
                    serial = values
                elif threads == 4:
                    case['thread_comparison'] = compare(serial, values)
                    assert case['thread_comparison']['bitwise_equal']
            report['cases'].append(case)
            args.report.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
            print(f'case {index+1}/12: passed', flush=True)
    report['petsc_loaded'] = sys.modules.get('petsc4py') is not None
    report['mpi_loaded'] = sys.modules.get('mpi4py') is not None
    report['diffusion_tests'] = []
    for threads in (1, 4):
        numba.set_num_threads(threads)
        suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'),
                                                  pattern='test_diffusion.py')
        with prototype(True, kernel):
            result = unittest.TextTestRunner(verbosity=1).run(suite)
        assert result.wasSuccessful()
        report['diffusion_tests'].append(dict(threads=threads, tests=result.testsRun, passed=True))
    args.report.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
