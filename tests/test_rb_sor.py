"""RB-SORの離散解、境界、同期、失敗検出を確認する。"""
from dataclasses import fields, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from streamer_drift_diffusion import SimulationConfig, initialize_simulation, poisson_calc, run_simulation
from streamer_drift_diffusion._acceleration import NUMBA_AVAILABLE
from streamer_drift_diffusion.electrostatics import _poisson_sweeps, calculate_electric_field
from streamer_drift_diffusion.__main__ import main


def equation(phi, state):
    """現行の境界条件を含む離散方程式A*phi（対角で規格化）。"""
    result = phi.copy()
    result[:, :-1] += (state.p1 * state.p2)[:, :-1] * phi[:, 1:]
    result[:, 1:] += (state.p1 * state.p5)[:, 1:] * phi[:, :-1]
    result[1:] += (state.p1 * state.p3)[1:] * phi[:-1]
    result[:-1] += (state.p1 * state.p4)[:-1] * phi[1:]
    result[-1] += (state.p1 * state.p4)[-1] * phi[-1]
    return result


class RedBlackTests(unittest.TestCase):
    def make_state(self, nr=9, nz=7):
        return initialize_simulation(SimulationConfig(nr=nr, nz=nz))

    def coefficients(self, state):
        return [getattr(state, f'p{i}') for i in range(1, 6)]

    def test_discrete_solution_and_field_on_odd_and_edge_grids(self):
        rng = np.random.default_rng(20261030)
        for shape in ((2, 3), (5, 7), (8, 5)):
            state = self.make_state(*shape)
            expected = rng.normal(size=shape) * 100.
            right_hand_side = equation(expected, state)
            rho = right_hand_side / state.p1
            for omega in (1., 1.5, 1.9):
                with self.subTest(shape=shape, omega=omega):
                    phi = np.zeros(shape)
                    poisson_calc(phi, rho, *self.coefficients(state), omega=omega,
                                 tolerance=1e-12, max_iterations=20000, verbose=False)
                    assert_allclose(phi, expected, rtol=0., atol=1e-7)
                    self.assertLess(np.max(np.abs(equation(phi, state)-right_hand_side)),
                                    np.max(np.abs(right_hand_side)) * 1e-10)
                    reference = self.make_state(*shape)
                    reference.phi[:] = expected
                    state.phi[:] = phi
                    config = SimulationConfig(nr=shape[0], nz=shape[1])
                    calculate_electric_field(reference, config)
                    calculate_electric_field(state, config)
                    assert_allclose(state.ex, reference.ex, rtol=0., atol=1e-6)
                    assert_allclose(state.ey, reference.ey, rtol=0., atol=1e-6)

    def test_chunk_boundaries_preserve_iteration_and_scale(self):
        state = self.make_state()
        args = (np.ones_like(state.phi), *self.coefficients(state), 1.5, -1.)
        whole, split = state.phi.copy(), state.phi.copy()
        count, scale, converged = _poisson_sweeps(whole, *args, 1e-10, 12)
        first = _poisson_sweeps(split, *args, 1e-10, 5)
        second = _poisson_sweeps(split, *args, first[1], 7)
        assert_array_equal(whole, split)
        self.assertEqual(count, first[0]+second[0])
        self.assertEqual(scale, second[1])
        self.assertFalse(converged)

    def test_zero_source_finishes_in_one_iteration(self):
        state = self.make_state()
        self.assertEqual(poisson_calc(state.pphi, state.rho, *self.coefficients(state),
                                      verbose=False, max_iterations=2), 1)
        assert_array_equal(state.pphi, 0.)

    def test_nonfinite_source_raises_with_and_without_iteration_limit(self):
        state = self.make_state()
        for bad in (np.nan, np.inf):
            rho = np.zeros_like(state.rho)
            rho[1, 2] = bad
            for limit in (None, 2):
                with self.subTest(bad=bad, limit=limit), self.assertRaisesRegex(RuntimeError, 'nonfinite'):
                    with np.errstate(invalid='ignore'):
                        poisson_calc(np.zeros_like(rho), rho, *self.coefficients(state),
                                     max_iterations=limit, verbose=False)

    def test_thread_setting_validation(self):
        for threads in (0, -1, True, 1.5, None):
            with self.subTest(threads=threads), self.assertRaisesRegex(ValueError, 'num_threads'):
                run_simulation(SimulationConfig(nr=3, nz=3, num_threads=threads),
                               write_output=False, verbose=False)

    def test_cli_uses_config_defaults_and_thread_override(self):
        defaults = SimulationConfig(num_threads=1, voltage=123., dt=2e-12)
        with patch('streamer_drift_diffusion.__main__.SimulationConfig',
                   side_effect=lambda **kwargs: replace(defaults, **kwargs)):
            for arguments, expected in (([], 1), (['--num-threads', '2'], 2)):
                with patch('streamer_drift_diffusion.__main__.run_simulation') as run:
                    self.assertEqual(main(arguments + ['--no-output', '--quiet']), 0)
                    passed = run.call_args.args[0]
                    self.assertEqual(passed.num_threads, expected)
                    self.assertEqual(passed.voltage, defaults.voltage)
                    self.assertEqual(passed.dt, defaults.dt)

    def test_thread_setting_is_applied_before_poisson_and_recorded(self):
        from streamer_drift_diffusion._acceleration import get_num_threads, set_num_threads
        from streamer_drift_diffusion.simulation import _prepare_adaptive
        previous_threads = get_num_threads()
        try:
            for threads in (1, 2):
                config = SimulationConfig(nr=3, nz=3, num_threads=threads,
                                          adaptive_timestep=True, t_end=0., voltage=0.)
                actual_threads = threads if NUMBA_AVAILABLE else 1

                def prepare(*args):
                    self.assertEqual(get_num_threads(), actual_threads)
                    return _prepare_adaptive(*args)

                with tempfile.TemporaryDirectory() as directory:
                    with patch('streamer_drift_diffusion.simulation._prepare_adaptive',
                               side_effect=prepare) as prepared:
                        run_simulation(config, output_dir=directory, verbose=False)
                        self.assertTrue(prepared.called)
                    metadata = json.loads((Path(directory) / 'run_config.json').read_text())
                    self.assertEqual(metadata['config']['num_threads'], threads)
                    self.assertEqual(metadata['num_threads_used'], actual_threads)
                    self.assertEqual(metadata['numba_available'], NUMBA_AVAILABLE)
        finally:
            set_num_threads(previous_threads)

    @unittest.skipUnless(NUMBA_AVAILABLE, 'Numbaのスレッド上限を確認する')
    def test_thread_limit_is_checked_before_output(self):
        from numba import config as numba_config
        config = SimulationConfig(nr=3, nz=3, num_threads=numba_config.NUMBA_NUM_THREADS+1)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'not_created'
            with self.assertRaises(ValueError):
                run_simulation(config, output_dir=output, verbose=False)
            self.assertFalse(output.exists())

    @unittest.skipUnless(NUMBA_AVAILABLE, 'Numbaなしでは同じRB更新を逐次実行する')
    def test_thread_counts_match_in_poisson_and_short_simulations(self):
        from numba import config as numba_config, get_num_threads, set_num_threads
        if numba_config.NUMBA_NUM_THREADS < 2:
            self.skipTest('2スレッドを使用できる環境で確認する')
        previous_threads = get_num_threads()
        state = self.make_state()
        rho = np.random.default_rng(7).normal(size=state.phi.shape)
        answers = []
        try:
            for threads in (1, 2):
                set_num_threads(threads)
                phi = np.zeros_like(rho)
                result = _poisson_sweeps(phi, rho, *self.coefficients(state), 1.5, -1., 1e-10, 37)
                simulations = []
                for adaptive in (False, True):
                    config = SimulationConfig(nr=9, nz=7, voltage=500., nmax=1e17,
                        z0=40e-6, cathode_radius=30e-6, max_step=2,
                        adaptive_timestep=adaptive, t_end=3e-12, num_threads=threads,
                        electrode_gap=40e-6, electron_diffusion=.1)
                    simulations.append(run_simulation(config, write_output=False, verbose=False))
                answers.append((phi, result, simulations))
        finally:
            set_num_threads(previous_threads)
        assert_array_equal(answers[0][0], answers[1][0])
        self.assertEqual(answers[0][1], answers[1][1])
        for left, right in zip(answers[0][2], answers[1][2]):
            for field in fields(left):
                if isinstance(getattr(left, field.name), np.ndarray):
                    assert_array_equal(getattr(left, field.name), getattr(right, field.name))


if __name__ == '__main__':
    unittest.main()
