"""FiPy Poissonの境界、独立した直接解、残差・異常系、設定と出力。"""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse.linalg import splu

from streamer_drift_diffusion import SimulationConfig, initialize_simulation, run_simulation
from streamer_drift_diffusion.electrostatics import calculate_potential
from streamer_drift_diffusion.timestepping import TimestepError
from streamer_drift_diffusion.__main__ import main
from poisson_reference import dense_poisson


class PoissonTests(unittest.TestCase):
    def config(self, **changes):
        return replace(SimulationConfig(nr=5, nz=7, delta=.1, electrode_gap=.3,
            cathode_radius=.2, voltage=500., nmax=0., background_ionization=0.,
            t_end=1e-12, poisson_tolerance=1e-12), **changes)

    def test_dense_reference_with_odd_even_grids_and_both_voltage_signs(self):
        for nr, nz in ((5, 7), (6, 8), (5, 8)):
            for voltage in (-500., 500.):
                with self.subTest(shape=(nr, nz), voltage=voltage):
                    state = initialize_simulation(self.config(nr=nr, nz=nz))
                    charge = np.random.default_rng(72).normal(size=(nr, nz))*1e3*state.gas_mask
                    matrix, rhs = dense_poisson(state, voltage, charge)
                    expected = np.linalg.solve(matrix, rhs).reshape((nr, nz))
                    answer = state.poisson.solve(charge, voltage)
                    assert_allclose(answer, expected, rtol=1e-11, atol=1e-11)
                    assert_array_equal(answer[~state.gas_mask], voltage)
                    self.assertLessEqual(state.poisson.residual, 1e-12)

    def test_laplace_maximum_principle_and_zero_solution(self):
        state = initialize_simulation(self.config())
        zero = np.zeros_like(state.ne)
        assert_array_equal(state.poisson.solve(zero, 0.), zero)
        for voltage in (-500., 500.):
            answer = state.poisson.solve(zero, voltage)/voltage
            self.assertGreaterEqual(answer.min(), -1e-12)
            self.assertLessEqual(answer.max(), 1.+1e-12)

    def test_small_charge_change_is_not_skipped_after_large_boundary_voltage(self):
        state = initialize_simulation(self.config(poisson_tolerance=1e-6))
        zero = np.zeros_like(state.ne)
        before = state.poisson.solve(zero, 500.).copy()
        charge = 1e-3*state.gas_mask
        after = state.poisson.solve(charge, 500.)
        matrix, rhs = dense_poisson(state, 0., charge)
        expected = np.linalg.solve(matrix, rhs).reshape(state.ne.shape)
        self.assertGreater(expected.max(), 1e-7)
        assert_allclose(after-before, expected, rtol=1e-5, atol=1e-11)

    def test_metal_charge_is_excluded(self):
        state = initialize_simulation(self.config())
        expected = state.poisson.solve(np.zeros_like(state.ne), 500.).copy()
        charge = np.where(state.gas_mask, 0., 1e99)
        assert_allclose(state.poisson.solve(charge, 500.), expected, rtol=1e-14)

    def test_solver_reuses_mesh_and_returns_live_view_with_explicit_snapshots(self):
        state = initialize_simulation(self.config())
        self.assertIs(state.poisson.potential.mesh, state.grid.mesh)
        first = state.poisson.solve(np.zeros_like(state.ne), 500.)
        saved = first.copy()
        state.poisson.solve(np.zeros_like(state.ne), -500.)
        assert_allclose(first, state.phi, rtol=0., atol=0.)
        assert_allclose(first, -saved, rtol=1e-14, atol=1e-12)
        self.assertTrue(np.shares_memory(first, state.phi_var.value))

    def test_nonfinite_inputs_and_failed_solve_do_not_commit_potential(self):
        config = self.config()
        state = initialize_simulation(config)
        calculate_potential(state, config)
        original = state.phi.copy()
        for bad in (np.nan, np.inf):
            with self.subTest(bad=bad), self.assertRaises(RuntimeError):
                state.poisson.solve(np.full_like(state.ne, bad), 500.)
            with self.assertRaises(RuntimeError):
                state.poisson.solve(state.rho, bad)
        state.poisson._lu = SimpleNamespace(solve=lambda rhs: np.zeros_like(rhs))
        with self.assertRaisesRegex(RuntimeError, 'residual'):
            calculate_potential(state, config)
        assert_array_equal(state.phi, original)
        self.assertIsNone(state.poisson._lu)
        calculate_potential(state, config)
        assert_allclose(state.phi, original, rtol=1e-14)

    def test_matrix_and_factorization_are_reused_for_changing_sources_and_boundaries(self):
        state = initialize_simulation(self.config())
        operator = state.poisson
        with patch.object(operator.equation, 'justResidualVector', wraps=operator.equation.justResidualVector) as assemble:
            with patch('streamer_drift_diffusion.electrostatics.splu', wraps=splu) as factorize:
                matrix = factor = None
                for voltage in (0., 500., -500., 0.):
                    charge = np.random.default_rng(72).normal(size=state.ne.shape)*1e3*state.gas_mask
                    state.radial_boundary[:] *= .9
                    state.upper_boundary[:] *= .9
                    expected_matrix, rhs = dense_poisson(state, voltage, charge)
                    answer = operator.solve(charge, voltage)
                    expected = np.linalg.solve(expected_matrix, rhs).reshape(state.ne.shape)
                    assert_allclose(answer, expected, rtol=1e-11, atol=1e-11)
                    assert_allclose(operator.equation.RHSvector,
                                    operator.source.value*state.volume.ravel(order='F'), rtol=0., atol=0.)
                    if matrix is None:
                        matrix, factor = operator._matrix, operator._lu
                    self.assertIs(operator._matrix, matrix)
                    self.assertIs(operator._lu, factor)
                self.assertEqual(assemble.call_count, 1)
                self.assertEqual(factorize.call_count, 1)
        assert_array_equal(operator.solve(np.zeros_like(state.ne), 0.), 0.)
        self.assertIs(operator._lu, factor)

    def test_close_releases_factorization_and_rebuilds_without_reassembling(self):
        state = initialize_simulation(self.config())
        operator = state.poisson
        with patch.object(operator.equation, 'justResidualVector', wraps=operator.equation.justResidualVector) as assemble:
            with patch('streamer_drift_diffusion.electrostatics.splu', wraps=splu) as factorize:
                expected = operator.solve(state.rho, 500.).copy()
                matrix, factor = operator._matrix, operator._lu
                operator.close()
                operator.close()
                self.assertIsNone(operator._lu)
                assert_allclose(operator.solve(state.rho, 500.), expected, rtol=0., atol=0.)
                self.assertIsNot(operator._lu, factor)
                self.assertIs(operator._matrix, matrix)
                self.assertEqual(assemble.call_count, 1)
                self.assertEqual(factorize.call_count, 2)

    def test_failed_initial_factorization_restores_potential_and_can_retry(self):
        state = initialize_simulation(self.config())
        state.phi_var.setValue(7.)
        original = state.phi.copy()
        with patch('streamer_drift_diffusion.electrostatics.splu', side_effect=RuntimeError('factorization failure')):
            with self.assertRaisesRegex(RuntimeError, 'factorization failure'):
                state.poisson.solve(state.rho, 500.)
        assert_array_equal(state.phi, original)
        self.assertIsNone(state.poisson._lu)
        matrix, rhs = dense_poisson(state, 500., state.rho)
        assert_allclose(state.poisson.solve(state.rho, 500.),
                        np.linalg.solve(matrix, rhs).reshape(state.ne.shape), rtol=1e-11, atol=1e-11)

    def test_nonfinite_solution_is_rejected_and_factorization_can_be_rebuilt(self):
        config = self.config()
        state = initialize_simulation(config)
        calculate_potential(state, config)
        original = state.phi.copy()
        for bad in (np.nan, np.inf):
            state.poisson._lu = SimpleNamespace(solve=lambda rhs, value=bad: np.full_like(rhs, value))
            with self.assertRaisesRegex(RuntimeError, 'residual'):
                calculate_potential(state, config)
            assert_array_equal(state.phi, original)
            self.assertIsNone(state.poisson._lu)
            calculate_potential(state, config)
            assert_allclose(state.phi, original, rtol=1e-14)

    def test_time_loop_releases_factorization_after_success_and_failure(self):
        config = self.config(t_end=3e-12)
        state = run_simulation(config, write_output=False, verbose=False)
        self.assertIsNone(state.poisson._lu)
        self.assertIsNotNone(state.poisson._matrix)
        calculate_potential(state, config)
        self.assertIsNotNone(state.poisson._lu)
        state.poisson.close()
        failed = initialize_simulation(config)
        with patch('streamer_drift_diffusion.simulation.initialize_simulation', return_value=failed):
            with self.assertRaisesRegex(TimestepError, 'max_steps'):
                run_simulation(replace(config, max_steps=1), write_output=False, verbose=False)
        self.assertIsNone(failed.poisson._lu)

    def test_cli_tolerance_and_fipy_metadata(self):
        defaults = self.config(poisson_tolerance=2e-10)
        with patch('streamer_drift_diffusion.__main__.SimulationConfig', return_value=defaults):
            with patch('streamer_drift_diffusion.__main__.run_simulation') as run:
                main(['--no-output'])
                self.assertEqual(run.call_args.args[0].poisson_tolerance, 2e-10)
        with patch('streamer_drift_diffusion.__main__.run_simulation') as run:
            main(['--poisson-tolerance', '1e-9', '--no-output'])
            self.assertEqual(run.call_args.args[0].poisson_tolerance, 1e-9)
        with tempfile.TemporaryDirectory() as directory:
            run_simulation(replace(defaults, adaptive_timestep=True, t_end=1e-12),
                           output_dir=directory, verbose=False)
            metadata = json.loads((Path(directory)/'run_config.json').read_text(encoding='utf-8'))
            self.assertEqual(metadata['poisson_backend'], 'fipy')
            self.assertEqual(metadata['poisson_solver'], 'SuperLU')
            self.assertEqual(metadata['poisson_solver_suite'], 'scipy')
            self.assertTrue(metadata['poisson_matrix_reused'])
            self.assertTrue(metadata['poisson_factorization_reused'])
            self.assertNotIn('numba_available', metadata)


if __name__ == '__main__':
    unittest.main()
