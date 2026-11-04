"""FiPy輸送の幾何、独立した流束解、係数更新、数値例外と試行復元。"""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from streamer_drift_diffusion import SimulationConfig, initialize_simulation
from streamer_drift_diffusion.grid import initialize_grid
from streamer_drift_diffusion.reactions import apply_reactions
from streamer_drift_diffusion.timestepping import try_density_step
from streamer_drift_diffusion.transport import ElectronTransport


def gas_state(nr=3, nz=4, spacing=1.):
    config = SimulationConfig(nr=nr, nz=nz, delta=spacing, electrode_gap=100.*spacing)
    grid = initialize_grid(config)
    return SimpleNamespace(**grid, ne=np.zeros((nr, nz)), de=np.zeros((nr, nz)),
                           vex=np.zeros((nr+1, nz)), vey=np.zeros((nr, nz+1)))


class FiPyTransportTests(unittest.TestCase):
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
                                  voltage=0., background_ionization=0., electron_diffusion=.5)
        state = initialize_simulation(config)
        state.ne[:] = np.random.default_rng(7).random(state.ne.shape)*state.gas_mask*1e15
        state.vex[:] = 500.
        state.vey[:] = -700.
        reference = deepcopy(state)
        operator = ElectronTransport(state, config.delta)
        original = state.ne.copy()
        calls = []

        def fail_first(current, current_config, *, dt):
            apply_reactions(current, current_config, dt=dt)
            calls.append(dt)
            if len(calls) == 1:
                current.ne[current.gas_mask] = -1.

        with patch('streamer_drift_diffusion.reactions.apply_reactions', side_effect=fail_first):
            actual, retries = try_density_step(state, config, config.dt, transport=operator)
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
        operator = ElectronTransport(state, 1.)

        def corrupt(**kwargs):
            operator.density.setValue(np.nan)

        with patch.object(operator.equation, 'solve', side_effect=corrupt):
            with self.assertRaisesRegex(FloatingPointError, 'FiPy transport'):
                operator.step(state, .1)
        assert_array_equal(state.ne, 1.)
        # 失敗した内部変数を、次の試行ではstateから復元する。
        assert_array_equal(operator.step(state, .05), 0.)
        assert_allclose(state.ne, 1., rtol=1e-14)

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
