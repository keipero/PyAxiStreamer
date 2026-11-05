"""FiPy Poissonの境界、独立した直接解、残差・異常系、設定と出力。"""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import fipy
from numpy.testing import assert_allclose, assert_array_equal

from streamer_drift_diffusion import SimulationConfig, initialize_simulation, run_simulation
from streamer_drift_diffusion.electrostatics import calculate_potential, PoissonSolver
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
        with patch.object(state.poisson.solver, '_solve', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'residual'):
                calculate_potential(state, config)
        assert_array_equal(state.phi, original)
        calculate_potential(state, config)
        assert_allclose(state.phi, original, rtol=1e-14)

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
            self.assertEqual(metadata['poisson_solver'], 'CachedLUSolver' if fipy.solvers.solver_suite == 'petsc' else 'LinearLUSolver')
            self.assertEqual(metadata['poisson_matrix_reuse'], fipy.solvers.solver_suite == 'petsc')
            self.assertNotIn('numba_available', metadata)
            self.assertEqual(metadata['fipy_solver_suite'], fipy.solvers.solver_suite)
            if fipy.solvers.solver_suite == 'petsc':
                from petsc4py import PETSc
                self.assertEqual(metadata['petsc_version'], '.'.join(map(str, PETSc.Sys.getVersion())))
                self.assertEqual(metadata['petsc_scalar_dtype'], 'float64')
                self.assertEqual(metadata['transport_default_pc'], 'jacobi')
                self.assertIn('mpi_library', metadata)
                self.assertEqual(metadata['poisson_options_prefix'], 'poisson_')
            else:
                self.assertNotIn('petsc_version', metadata)

    def test_repeated_sources_and_voltage_zero_return_match_dense_reference(self):
        """同じ行列で電圧・電荷を変え、電圧0への復帰も独立解と一致させる。"""
        state = initialize_simulation(self.config())
        operator = state.poisson
        self.addCleanup(operator.close)
        with patch.object(operator.equation, 'solve', wraps=operator.equation.solve) as assembly:
            for voltage, scale in ((500., 0.), (-500., 1.), (0., -2.), (0., 0.), (500., .01)):
                rho = np.random.default_rng(31).normal(size=state.ne.shape)*scale
                matrix, rhs = dense_poisson(state, voltage, rho)
                expected = np.linalg.solve(matrix, rhs).reshape(state.ne.shape)
                assert_allclose(operator.solve(rho, voltage), expected, rtol=1e-11, atol=1e-11)
            self.assertEqual(assembly.call_count, 1 if operator.reuse_matrix else 5)

    @unittest.skipUnless(fipy.solvers.solver_suite == 'petsc', 'PETSc cache')
    def test_lu_factor_is_reused_and_close_allows_new_factor_without_reassembly(self):
        """右辺変更でKSPとLU因子を保持し、明示解放後も同じ行列で再開できる。"""
        state = initialize_simulation(self.config())
        operator = state.poisson
        self.addCleanup(operator.close)
        with patch.object(operator.equation, 'solve', wraps=operator.equation.solve) as assembly:
            operator.solve(state.rho, 500.)
            ksp = operator.solver.ksp
            factor = ksp.getPC().getFactorMatrix().handle
            operator.solve(state.rho+1., -500.)
            self.assertIs(operator.solver.ksp, ksp)
            self.assertEqual(operator.solver.ksp.getPC().getFactorMatrix().handle, factor)
            operator.close()
            operator.close()
            self.assertFalse(ksp)
            self.assertFalse(hasattr(operator.solver, '_globalVectors'))
            operator.solve(state.rho, 500.)
            self.assertIsNot(operator.solver.ksp, ksp)
            self.assertEqual(assembly.call_count, 1)

    @unittest.skipUnless(fipy.solvers.solver_suite == 'petsc', 'PETSc cache')
    def test_cached_backend_exception_and_nonfinite_solution_restore_and_recover(self):
        """再利用中の例外・非有限解で正本を復元し、作業領域を破棄して回復する。"""
        state = initialize_simulation(self.config())
        operator = state.poisson
        self.addCleanup(operator.close)
        for failure in ('exception', 'nonfinite'):
            operator.solve(state.rho, 500.)
            before = state.phi.copy()
            ksp = operator.solver.ksp

            def fail(matrix, solution, rhs):
                if failure == 'exception':
                    raise RuntimeError('cached failure')
                solution.set(np.nan)
                return solution

            with patch.object(operator.solver, '_solve_', side_effect=fail):
                with self.assertRaises(RuntimeError):
                    operator.solve(state.rho, -500.)
            assert_array_equal(state.phi, before)
            self.assertFalse(ksp)
            self.assertFalse(hasattr(operator.solver, '_globalVectors'))
            assert_allclose(operator.solve(state.rho, 500.), before, rtol=1e-14)

    @unittest.skipUnless(fipy.solvers.solver_suite == 'petsc', 'PETSc cache')
    def test_prefixed_options_failure_on_first_setup_releases_and_recovers(self):
        """初回設定のPETScエラーを復元し、接頭辞付きオプションを修正すると再開できる。"""
        from petsc4py import PETSc
        options = PETSc.Options()
        key = 'poisson_pc_type'
        previous = options.getAll().get(key)
        state = initialize_simulation(self.config())
        operator = state.poisson
        self.addCleanup(operator.close)
        state.phi_var.setValue(17.)
        try:
            options[key] = 'invalid_pc_for_test'
            with self.assertRaisesRegex(RuntimeError, 'PETSc Poisson'):
                operator.solve(state.rho, 500.)
            assert_array_equal(state.phi, 17.)
            self.assertFalse(hasattr(operator.solver, 'ksp'))
            self.assertFalse(hasattr(operator.solver, '_globalVectors'))
            options[key] = 'lu'
            operator.solve(state.rho, 500.)
            self.assertEqual(operator.solver.ksp.getOptionsPrefix(), 'poisson_')
            self.assertEqual(operator.solver.ksp.getPC().type, 'lu')
        finally:
            if previous is None:
                del options[key]
            else:
                options[key] = previous

    def test_rebuild_option_keeps_same_physics_and_run_releases_resources(self):
        """比較用の再組立モードと既定解が一致し、時間ループ終了で分解を解放する。"""
        config = self.config()
        state = initialize_simulation(config)
        rebuild = PoissonSolver(state.grid, state.gas_mask, config, reuse_matrix=False)
        self.addCleanup(state.poisson.close)
        self.addCleanup(rebuild.close)
        for voltage in (500., -500.):
            assert_allclose(state.poisson.solve(state.rho+1., voltage), rebuild.solve(state.rho+1., voltage), rtol=1e-13)
        completed = run_simulation(config, write_output=False, verbose=False)
        self.assertFalse(hasattr(completed.poisson.solver, 'ksp'))
        assert_allclose(completed.poisson.solve(completed.rho, config.voltage), completed.phi)
        completed.poisson.close()

    @unittest.skipUnless(fipy.solvers.solver_suite == 'petsc', 'PETSc cache')
    def test_standalone_solver_collection_releases_native_ksp(self):
        """単独使用で明示closeを忘れても、回収時にPETScのKSPを解放する。"""
        import gc
        import weakref
        state = initialize_simulation(self.config())
        state.poisson.solve(state.rho, 500.)
        reference = weakref.ref(state.poisson.solver)
        ksp = state.poisson.solver.ksp
        del state
        gc.collect()
        self.assertIsNone(reference())
        self.assertFalse(ksp)


if __name__ == '__main__':
    unittest.main()
