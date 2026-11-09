"""FiPy輸送の幾何、独立した流束解、係数更新、数値例外と試行復元。"""
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from streamer_drift_diffusion import SimulationConfig, initialize_simulation, run_simulation
from streamer_drift_diffusion.reactions import apply_reactions
from streamer_drift_diffusion.timestepping import TimestepError, advance_density, timestep_limits
from streamer_drift_diffusion.transport import ElectronTransport


def gas_state(nr=3, nz=4, spacing=1.):
    config = SimulationConfig(nr=nr, nz=nz, delta=spacing, electrode_gap=100.*spacing,
                              voltage=0., nmax=0., background_ionization=0.)
    return initialize_simulation(config)


class FiPyTransportTests(unittest.TestCase):
    def test_numba_matches_fipy_with_mixed_fluxes_and_electrode(self):
        rng = np.random.default_rng(342)
        jit_state = gas_state(7, 9, .3)
        reference = gas_state(7, 9, .3)
        for state in (jit_state, reference):
            state.gas_mask[1, 3:5] = False
            state.gas_mask[5:, 7:] = False
        jit = ElectronTransport(jit_state, .3)
        fipy = ElectronTransport(reference, .3, backend="fipy")
        for sign in (1., -1.):
            initial = np.exp(rng.uniform(-4., 4., jit_state.ne.shape))*jit_state.gas_mask
            velocity_r = sign*rng.uniform(-2., 2., jit_state.vex.shape)
            velocity_z = sign*rng.uniform(-2., 2., jit_state.vey.shape)
            diffusion = rng.uniform(0., .4, jit_state.ne.shape)
            diffusion[::3] = 0.
            for state in (jit_state, reference):
                state.ne[:] = initial
                state.vex[:], state.vey[:], state.de[:] = velocity_r, velocity_z, diffusion
            jit.prepare(jit_state)
            fipy.prepare(reference)
            for name in ("advection_rates", "diffusion_rates", "boundary_rates",
                         "transport_limit", "diffusion_limit"):
                assert_allclose(getattr(jit, name), getattr(fipy, name), rtol=1e-14, atol=0.)
            dt = .95*jit.transport_limit
            jit_loss = jit.step(jit_state, dt, prepared=True)
            fipy_loss = fipy.step(reference, dt, prepared=True)
            assert_allclose(jit_state.ne, reference.ne, rtol=3e-14, atol=1e-15)
            assert_allclose(jit_loss, fipy_loss, rtol=1e-14)
            assert_array_equal(jit_state.ne[~jit_state.gas_mask], 0.)
            self.assertGreaterEqual(jit_state.ne.min(), 0.)
            final = 2*np.pi*np.sum(jit_state.ne*jit_state.volume)
            assert_allclose(final + jit_loss.sum(), 2*np.pi*np.sum(initial*jit_state.volume),
                            rtol=1e-14)

    def test_prepared_limits_bound_actual_fipy_update(self):
        """合算CFLの上界と純拡散CFLの対角を、単位密度から検査する。"""
        state = gas_state()
        state.gas_mask[1, 2] = False
        rng = np.random.default_rng(92)
        state.de[:] = rng.random(state.ne.shape)
        velocity_r = rng.uniform(-2., 2., state.vex.shape)
        velocity_z = rng.uniform(-2., 2., state.vey.shape)
        operator = ElectronTransport(state, 1.)
        for mixed in (False, True):
            with self.subTest(mixed=mixed):
                state.vex[:] = velocity_r if mixed else 0.
                state.vey[:] = velocity_z if mixed else 0.
                operator.prepare(state)
                dt = .1*operator.transport_limit
                diagonal = []
                for index in zip(*np.nonzero(state.gas_mask)):
                    state.ne[:] = 0.
                    state.ne[index] = 1.
                    initial = 2*np.pi*state.volume[index]
                    losses = operator.step(state, dt, prepared=True)
                    diagonal.append((1.-state.ne[index])/dt)
                    final = 2*np.pi*np.sum(state.ne*state.volume)
                    self.assertAlmostEqual((final+losses.sum())/initial, 1., places=13)
                ratio = operator.transport_limit*max(diagonal)
                if mixed:
                    self.assertLess(ratio, 1.)
                    self.assertGreaterEqual(ratio, .5)
                else:
                    self.assertAlmostEqual(ratio, 1., places=13)
                    self.assertEqual(operator.transport_limit, operator.diffusion_limit)

    def test_van_leer_axial_flux_uses_gas_slope_next_to_electrode(self):
        """電極隣接セルも気体側の線形勾配で再構築する。"""
        for electrode in (False, True):
            with self.subTest(electrode=electrode):
                state = gas_state(3, 5)
                state.ne[1] = [1., 2., 3., 4., 5.]
                if electrode:
                    state.gas_mask[1, 1] = False
                    state.ne[1, 1] = 0.
                state.vey[1, 3] = 1.
                losses = ElectronTransport(state, 1.).step(state, .2)
                # 気体内の勾配は1。面値は3 + (1 - CFL)/2。
                transfer = .2 * 3.4
                assert_allclose(state.ne[1, 2:4], [3. - transfer, 4. + transfer],
                                rtol=1e-14)
                assert_array_equal(losses, 0.)

    def test_van_leer_uses_axis_symmetry_on_first_radial_face(self):
        state = gas_state()
        state.ne[:, 1] = [1., 2., 4.]
        state.vex[1, 1] = 1.
        losses = ElectronTransport(state, 1.).step(state, .2)
        # r=0.5での勾配は0.5、セル体積0.5、面積1、面速度1。
        transfer = .2 * (1. + .5*.5*(.5 - .2))
        assert_allclose(state.ne[:2, 1], [1. - transfer/.5, 2. + transfer/1.5],
                        rtol=1e-14)
        assert_array_equal(losses, 0.)

    def test_radial_transport_near_cfl_keeps_density_and_particle_balance(self):
        state = gas_state(5, 5)
        state.gas_mask[3, 2] = False
        state.ne[:] = np.array([100., 1., .01, 0., 0.])[:, None]
        state.vex[1:3] = -1.
        state.vey[:, -1] = .25
        state.de[:] = .01
        operator = ElectronTransport(state, 1.).prepare(state)
        initial = 2*np.pi*np.sum(state.ne*state.volume)
        losses = operator.step(state, .99*operator.transport_limit, prepared=True)
        self.assertTrue(np.all(state.ne >= 0.))
        self.assertGreater(losses.sum(), 0.)
        final = 2*np.pi*np.sum(state.ne*state.volume)
        assert_allclose(final + losses.sum(), initial, rtol=1e-14)

    def test_fixed_trial_rejects_unsafe_pure_advection_step(self):
        config = SimulationConfig(nr=3, nz=5, delta=1., electrode_gap=100.,
                                  voltage=0., adaptive_timestep=False)
        state = initialize_simulation(config)
        state.ne[:] = 1.
        state.vey[:, 2] = 1.
        operator = ElectronTransport(state, 1.).prepare(state)
        with self.assertRaisesRegex(TimestepError, "explicit drift-diffusion constraint"):
            advance_density(state, config, 1.1*operator.transport_limit,
                            transport=operator, prepared=True)
        assert_array_equal(state.ne, 1.)
        assert_array_equal(state.electron_losses, 0.)

    def test_cfl_and_retried_transport_share_one_preparation(self):
        """CFL評価後の棄却・再試行でも、同じ係数と復元した旧密度を使う。"""
        config = SimulationConfig(nr=3, nz=4, delta=1., electrode_gap=100.,
                                  voltage=0., nmax=0., background_ionization=0.,
                                  electron_diffusion=.5, adaptive_timestep=True)
        state = initialize_simulation(config)
        reference = initialize_simulation(config)
        state.ne[:] = reference.ne[:] = 1.
        operator = ElectronTransport(state, config.delta)
        attempts = []

        def fail_once(current, current_config, *, dt):
            apply_reactions(current, current_config, dt=dt)
            attempts.append(dt)
            if len(attempts) == 1:
                current.ne[:] = -1.

        with patch.object(operator, 'prepare', wraps=operator.prepare) as prepared:
            limits = timestep_limits(state, config, transport=operator)
            with patch('streamer_drift_diffusion.reactions.apply_reactions', side_effect=fail_once):
                dt, retries = advance_density(state, config, limits.dt, transport=operator, prepared=True)
            self.assertEqual(prepared.call_count, 1)
        expected_loss = ElectronTransport(reference, config.delta).step(reference, dt)
        apply_reactions(reference, config, dt=dt)
        self.assertEqual(retries, 1)
        self.assertEqual(dt, limits.dt/2)
        assert_allclose(state.ne, reference.ne, rtol=1e-14)
        assert_allclose(state.electron_losses, expected_loss, rtol=1e-14)

    def test_time_loop_prepares_coefficients_once_per_update(self):
        """両モードの時間ループがCFL・輸送間で面係数を再生成しない。"""
        real_prepare = ElectronTransport.prepare
        for adaptive in (False, True):
            config = SimulationConfig(nr=2, nz=3, voltage=0., nmax=0.,
                background_ionization=0., adaptive_timestep=adaptive,
                t_end=3e-12, output_interval=1e-12)
            with self.subTest(adaptive=adaptive):
                with patch.object(ElectronTransport, 'prepare', autospec=True, side_effect=real_prepare) as prepared:
                    run_simulation(config, write_output=False, verbose=False)
                self.assertEqual(prepared.call_count, 3)

    def test_transport_reuses_shared_mesh_and_geometry_arrays(self):
        state = gas_state(5, 7, .1)
        with patch('streamer_drift_diffusion.grid.CylindricalGrid2D', side_effect=AssertionError('rebuilt mesh')):
            operator = ElectronTransport(state, .1)
        self.assertIs(operator.mesh, state.grid.mesh)
        self.assertIs(state.volume, state.grid.volume)
        self.assertTrue(np.shares_memory(state.sr, state.grid.areas[0]))
        self.assertTrue(np.shares_memory(state.sz, state.grid.areas[1]))

    def test_face_neighbors_have_coordinate_order_and_cover_each_face_once(self):
        state = gas_state(3, 5)
        grid = state.grid
        radial, axial = grid.face_ids
        assert_array_equal(np.sort(np.concatenate((radial.ravel(), axial.ravel()))),
                           np.arange(grid.mesh.numberOfFaces))
        for i, j in np.ndindex(radial.shape):
            self.assertEqual(grid.low[radial[i, j]], (i-1)+3*j if i else -1)
            self.assertEqual(grid.high[radial[i, j]], i+3*j if i < 3 else -1)
        for i, j in np.ndindex(axial.shape):
            self.assertEqual(grid.low[axial[i, j]], i+3*(j-1) if j else -1)
            self.assertEqual(grid.high[axial[i, j]], i+3*j if j < 5 else -1)
        values = np.arange(grid.mesh.numberOfFaces)
        assert_array_equal(grid.pack_faces(*grid.split_faces(values)), values)

    def test_shared_boundary_labels_and_distances_include_internal_electrode(self):
        state = gas_state()
        state.gas_mask[1, 2] = False
        grid = state.grid
        face = grid.face_ids[1][1, 2]
        self.assertEqual(grid.boundary_kind[face], 0)
        self.assertEqual(grid.distances(state.gas_mask)[face], .5)
        self.assertEqual(grid.distances(state.gas_mask)[grid.face_ids[1][0, 2]], 1.)
        assert_array_equal(grid.boundary_kind[grid.face_ids[1][:, 0]], 1)
        assert_array_equal(grid.boundary_kind[grid.face_ids[1][:, -1]], 2)
        assert_array_equal(grid.boundary_kind[grid.face_ids[0][-1]], 2)
        assert_array_equal(grid.area[grid.face_ids[0][0]], 0.)
        assert_array_equal(grid.sum_faces(np.ones(grid.mesh.numberOfFaces)), 4.)

    def test_cylindrical_cell_mapping_and_volume(self):
        state = gas_state(5, 7, .1)
        operator = ElectronTransport(state, .1)
        assert_allclose(operator.mesh.cellVolumes, state.volume.ravel(order='F'),
                        rtol=1e-14, atol=0.)
        assert_allclose(np.asarray(operator.mesh.cellCenters[0]).reshape((5, 7), order='F'),
                        np.broadcast_to(state.rh[:, None], (5, 7)), rtol=1e-14)
        assert_allclose(np.asarray(operator.mesh.cellCenters[1]).reshape((5, 7), order='F'),
                        np.broadcast_to(state.zh[None, :], (5, 7)), rtol=1e-14)

    def test_harmonic_diffusion_transfers_exact_cell_particle_number(self):
        state = gas_state()
        state.ne[1, 1] = 1.
        state.de[1, 1], state.de[1, 2] = 1., 9.
        losses = ElectronTransport(state, 1.).step(state, .1)
        expected = np.zeros_like(state.ne)
        expected[1, 1], expected[1, 2] = .82, .18
        assert_allclose(state.ne, expected, rtol=1e-14, atol=1e-16)
        assert_array_equal(losses, 0.)

    def test_half_cell_absorption_uses_gas_coefficient_not_metal_average(self):
        state = gas_state()
        state.gas_mask[1, 2] = False
        state.ne[1, 1] = 1.
        state.de[1, 1] = 1.
        losses = ElectronTransport(state, 1.).step(state, .1)
        # 他の気体のD=0で拡散を遮断。金属面だけへ2D/hで吸収。
        expected = np.zeros_like(state.ne)
        expected[1, 1] = .8
        assert_allclose(state.ne, expected, rtol=1e-14, atol=1e-16)
        assert_allclose(losses, [2*np.pi*.1*2*1.5, 0., 0.], rtol=1e-14)

    def test_cached_equation_follows_velocity_reversal_and_old_density(self):
        state = gas_state(2, 4)
        state.ne[:] = 1.
        operator = ElectronTransport(state, 1.)
        state.vey[:] = 1.
        first_loss = operator.step(state, .1)
        assert_allclose(state.ne, np.tile([.9, 1., 1., 1.], (2, 1)), rtol=1e-14)
        state.vey[:] = -1.
        second_loss = operator.step(state, .1)
        assert_allclose(state.ne, np.tile([.91, 1., 1., .9], (2, 1)), rtol=1e-14)
        assert_allclose(first_loss, [0., 0., 2*np.pi*.2], rtol=1e-14)
        assert_allclose(second_loss, [0., 2*np.pi*.18, 0.], rtol=1e-14)

    def test_cached_equation_follows_diffusion_changes_and_zero_barrier(self):
        state = gas_state()
        state.ne[1, 1] = 1.
        operator = ElectronTransport(state, 1.)
        state.de[1, 1] = 1.
        operator.step(state, .1)
        self.assertEqual(state.ne[1, 2], 0.)
        state.de[1, 2] = 9.
        operator.step(state, .1)
        self.assertAlmostEqual(state.ne[1, 2], .18)
        state.de[:] = 0.
        saved = state.ne.copy()
        operator.step(state, .025)
        assert_allclose(state.ne, saved, rtol=1e-14, atol=1e-16)

    def test_zero_faces_are_valid_under_strict_numpy_and_warning_settings(self):
        state = gas_state()
        state.ne[:] = np.arange(state.ne.size).reshape(state.ne.shape)
        saved = state.ne.copy()
        operator = ElectronTransport(state, 1.)
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                losses = operator.step(state, .1)
        assert_allclose(state.ne, saved, rtol=1e-14, atol=1e-16)
        assert_array_equal(losses, 0.)

    def test_failed_reaction_resynchronizes_fipy_old_values_and_losses(self):
        config = SimulationConfig(nr=5, nz=7, delta=1e-4, electrode_gap=3e-4,
                                  voltage=0., background_ionization=0., electron_diffusion=.5, adaptive_timestep=True)
        state = initialize_simulation(config)
        state.ne[:] = np.random.default_rng(7).random(state.ne.shape)*state.gas_mask*1e15
        state.vex[:] = 500.
        state.vey[:] = -700.
        reference = initialize_simulation(config)
        reference.ne[:] = state.ne
        reference.vex[:], reference.vey[:] = state.vex, state.vey
        operator = ElectronTransport(state, config.delta)
        original = state.ne.copy()
        calls = []

        def fail_first(current, current_config, *, dt):
            apply_reactions(current, current_config, dt=dt)
            calls.append(dt)
            if len(calls) == 1:
                current.ne[current.gas_mask] = -1.

        with patch('streamer_drift_diffusion.reactions.apply_reactions', side_effect=fail_first):
            actual, retries = advance_density(state, config, config.dt, transport=operator)
        expected_loss = ElectronTransport(reference, config.delta).step(reference, config.dt/2)
        apply_reactions(reference, config, dt=config.dt/2)
        self.assertEqual(retries, 1)
        self.assertEqual(actual, config.dt/2)
        assert_allclose(state.ne, reference.ne, rtol=1e-14)
        assert_allclose(state.electron_losses, expected_loss, rtol=1e-14)
        assert_array_equal(np.asarray(operator.density.old).reshape(state.ne.shape, order='F'), original)

    def test_nonfinite_solver_result_is_not_committed(self):
        state = gas_state()
        state.ne[:] = 1.
        operator = ElectronTransport(state, 1., backend="fipy")

        def corrupt(**kwargs):
            operator.density.setValue(np.nan)

        with patch.object(operator.equation, 'solve', side_effect=corrupt):
            with self.assertRaisesRegex(FloatingPointError, 'FiPy transport'):
                operator.step(state, .1)
        assert_array_equal(state.ne, 1.)
        # 失敗した内部変数を、次の試行ではstateから復元する。
        assert_array_equal(operator.step(state, .05), 0.)
        assert_allclose(state.ne, 1., rtol=1e-14)

    def test_nonfinite_numba_result_is_not_committed(self):
        state = gas_state()
        state.ne[:] = 1.
        operator = ElectronTransport(state, 1.)
        corrupt = (np.full(state.ne.size, np.nan), np.zeros(3))
        with patch('streamer_drift_diffusion.transport.transport_update', return_value=corrupt):
            with self.assertRaisesRegex(FloatingPointError, 'Numba transport'):
                operator.step(state, .1)
        assert_array_equal(state.ne, 1.)
        assert_array_equal(operator.step(state, .05), 0.)

    def test_invalid_velocity_and_geometry_fail_before_commit(self):
        state = gas_state()
        operator = ElectronTransport(state, 1.)
        state.vex[1, 1] = np.inf
        with self.assertRaises(FloatingPointError):
            operator.step(state, .1)
        assert_array_equal(state.ne, 0.)
        state.vex[:] = 0.
        state.gas_mask[1, 1] = False
        with self.assertRaisesRegex(ValueError, 'mask'):
            operator.step(state, .1)


if __name__ == '__main__':
    unittest.main()
