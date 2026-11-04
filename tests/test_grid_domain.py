"""電極を含む領域、Poisson境界、有限体積輸送の収支を検証する。"""
from dataclasses import replace
import csv
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from streamer_drift_diffusion import SimulationConfig, initialize_simulation, run_simulation
from streamer_drift_diffusion.electrostatics import calculate_charge_density, calculate_potential
from streamer_drift_diffusion.transport import ElectronTransport
from streamer_drift_diffusion.grid import initialize_grid
from poisson_reference import dense_poisson
from streamer_drift_diffusion.reactions import apply_reactions
from streamer_drift_diffusion.timestepping import transport_limit, chemistry_limit, dielectric_limit, advance_density
from streamer_drift_diffusion.simulation_types import gap_length
from streamer_drift_diffusion.simulation import prepare_step


class GridDomainTests(unittest.TestCase):
    def config(self, **kwargs):
        return replace(SimulationConfig(nr=12, nz=15, delta=1e-4, electrode_gap=8e-4,
            cathode_radius=3e-4, z0=6e-4, voltage=500., poisson_tolerance=1e-11,
            t_end=3e-12, background_ionization=0.), **kwargs)

    def test_gap_is_independent_of_domain_and_mask_tracks_shape(self):
        config = self.config()
        short = initialize_grid(config)
        long = initialize_grid(replace(config, nz=20))
        self.assertEqual(gap_length(config), gap_length(replace(config, nz=20)))
        assert_array_equal(short['gas_mask'], long['gas_mask'][:, :config.nz])
        self.assertTrue(np.any(~long['gas_mask']))
        self.assertTrue(np.any(long['gas_mask'][:, config.nz:]))

    def test_full_cylinder_volume_including_outermost_cell(self):
        config = self.config()
        grid = initialize_grid(config)
        expected = np.pi * (config.nr*config.delta)**2 * config.nz*config.delta
        self.assertAlmostEqual(np.sum(grid['volume']) * 2*np.pi / expected, 1.)
        assert_allclose(grid['sz'][-1, 0] * 2*np.pi,
                        np.pi*((config.nr*config.delta)**2-((config.nr-1)*config.delta)**2))

    def test_initial_and_reaction_densities_vanish_inside_electrode(self):
        config = self.config(background_ionization=1e13)
        state = initialize_simulation(config)
        apply_reactions(state, config, dt=config.dt/2)
        for name in ('ne', 'n2', 'o2', 'n2p', 'o2p', 'om', 'o2m'):
            assert_array_equal(getattr(state, name)[~state.gas_mask], 0.)
        assert_allclose(state.ne + state.om + state.o2m, state.n2p+state.o2p, rtol=1e-15)

    def test_poisson_matches_dense_direct_solution_with_electrode(self):
        config = self.config(nr=5, nz=7, electrode_gap=3e-4, z0=2e-4)
        state = initialize_simulation(config)
        state.rho[:] = np.random.default_rng(42).normal(size=state.phi.shape)*1e8*state.gas_mask
        matrix, rhs = dense_poisson(state, config.voltage, state.rho)
        expected = np.linalg.solve(matrix, rhs).reshape(state.phi.shape)
        calculate_potential(state, config, verbose=False)
        assert_allclose(state.phi, expected, rtol=1e-11, atol=1e-10)


    def test_electrode_voltage_stays_fixed_with_space_charge_and_both_polarities(self):
        for voltage in (-1000., 1000.):
            config = self.config(voltage=voltage)
            state = initialize_simulation(config)
            state.n2p[state.gas_mask] += 1e13
            calculate_charge_density(state, config)
            calculate_potential(state, config, verbose=False)
            assert_array_equal(state.phi[~state.gas_mask], voltage)
            self.assertLessEqual(state.poisson.residual, config.poisson_tolerance)

    def test_transport_conserves_electrons_including_all_boundary_losses(self):
        config = self.config()
        for sign in (-1., 1.):
            state = initialize_simulation(config)
            state.ne[:] = 1e18 * state.gas_mask
            state.vex[:] = 2e4
            state.vey[:] = sign*3e4
            dt = transport_limit(state, config)*.5
            initial = np.sum(state.ne*state.volume)*2*np.pi
            losses = np.zeros(3)
            transport = ElectronTransport(state, config.delta)
            for _ in range(100):
                losses += transport.step(state, dt)
            final = np.sum(state.ne*state.volume)*2*np.pi
            self.assertAlmostEqual((final+losses.sum())/initial, 1., places=13)
            self.assertTrue(np.all(state.ne >= 0.))
            self.assertTrue(np.all(losses >= 0.))
            self.assertGreater(losses[0] if sign > 0 else losses[1], 0.)
            self.assertGreater(losses[2], 0.)

    def test_zero_density_cannot_emit_from_electrode_or_outer_boundaries(self):
        config = self.config()
        state = initialize_simulation(config)
        for sign in (-1., 1.):
            state.ne[:] = 0.
            state.vex[:] = sign*1e3
            state.vey[:] = sign*2e3
            losses = ElectronTransport(state, config.delta).step(state, 1e-12)
            assert_array_equal(state.ne, 0.)
            assert_array_equal(losses, 0.)

    def test_chemistry_and_dielectric_limits_exclude_metal_cells(self):
        config = self.config()
        state = initialize_simulation(config)
        prepare_step(state, config, 0., verbose=False)
        state.ne[:] = 1e20*state.gas_mask
        state.k1[:] = 1e-18
        expected = chemistry_limit(state, config), dielectric_limit(state, config)
        for name in ('ne', 'n2', 'o2', 'k1', 'k2', 'k3'):
            getattr(state, name)[~state.gas_mask] = 1e40
        state.abs_e[~state.gas_mask] = 1e-40
        assert_allclose((chemistry_limit(state, config), dielectric_limit(state, config)), expected)

    def test_rejected_transport_does_not_accumulate_boundary_loss(self):
        config = self.config(adaptive_timestep=True)
        state = initialize_simulation(config)
        state.vex[:] = 1e5
        state.vey[:] = 1e5
        state.k1[:] = state.k2[:] = state.k3[:] = 0.
        initial = np.sum(state.ne*state.volume)*2*np.pi
        calls = []

        real_step = ElectronTransport.step

        def fail_first(operator, current, dt):
            losses = real_step(operator, current, dt)
            calls.append(1)
            if len(calls) == 1:
                state.ne[state.gas_mask] = -1.
            return losses

        with patch.object(ElectronTransport, 'step', autospec=True, side_effect=fail_first):
            dt, retries = advance_density(state, config, config.dt)
        self.assertEqual(retries, 1)
        self.assertEqual(dt, config.dt/2)
        final = np.sum(state.ne*state.volume)*2*np.pi
        self.assertAlmostEqual((final+state.electron_losses.sum())/initial, 1., places=13)

    def test_adaptive_output_contains_geometry_and_loss_history(self):
        config = self.config(adaptive_timestep=True, t_end=2e-12, voltage=-500., output_interval=1e-12)
        with tempfile.TemporaryDirectory() as directory:
            state = run_simulation(config, output_dir=directory, verbose=False)
            mask = np.loadtxt(Path(directory)/'gas_mask.csv').reshape(state.ne.shape)
            assert_array_equal(mask, state.gas_mask)
            with (Path(directory)/'timestep_history.csv').open() as history:
                rows = list(csv.DictReader(history))
            self.assertTrue(rows)
            self.assertIn('electrode_electron_loss', rows[-1])
            assert_array_equal(state.ne[~state.gas_mask], 0.)
            self.assertTrue(np.all(np.isfinite(state.phi)))

    def test_invalid_gap_and_grid_without_gas_fail(self):
        for gap in (0., -1., np.nan, np.inf):
            with self.assertRaises(ValueError):
                initialize_simulation(self.config(electrode_gap=gap))
        with self.assertRaisesRegex(ValueError, 'no gas'):
            initialize_simulation(self.config(electrode_gap=1e-15, cathode_radius=1.))

    def test_impulse_with_embedded_electrode(self):
        with tempfile.TemporaryDirectory() as directory:
            waveform = Path(directory)/'triangle.txt'
            waveform.write_text('0 0\n0.001 1\n0.002 0\n', encoding='utf-8')
            config = self.config(adaptive_timestep=True, t_end=3e-12,
                voltage_mode='impulse', impulse_peak_voltage=1000.,
                voltage_waveform_file=str(waveform))
            state = run_simulation(config, write_output=False, verbose=False)
            assert_array_equal(state.phi[~state.gas_mask], 0.)
            assert_array_equal(state.ne[~state.gas_mask], 0.)
            self.assertTrue(np.all(np.isfinite(state.abs_e)))
            self.assertTrue(np.all(state.ne >= 0.))


if __name__ == '__main__':
    unittest.main()
