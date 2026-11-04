"""明示的ドリフト拡散の面流束、非負性、収支と解析解を検証する。"""
from dataclasses import replace
import csv
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from streamer_drift_diffusion import SimulationConfig, initialize_simulation, run_simulation
from streamer_drift_diffusion.transport import ElectronTransport
from streamer_drift_diffusion.grid import initialize_grid
from streamer_drift_diffusion.timestepping import (
    TimestepError, diffusion_limit, transport_limit, timestep_limits, try_density_step,
)
from streamer_drift_diffusion.__main__ import main


def gaussian_average(grid, sigma, center):
    """無限空間のGaussianを円環セル体積で平均する。振幅は1。"""
    r, dr, z, dz = (grid[k] for k in ('r', 'dr', 'z', 'dz'))
    radial = 2*sigma**2*(np.exp(-r*r/(2*sigma*sigma))
                        - np.exp(-(r+dr)**2/(2*sigma*sigma)))/((r+dr)**2-r*r)
    erf = np.vectorize(math.erf)
    axial = sigma*np.sqrt(np.pi/2)/dz*(erf((z+dz-center)/(np.sqrt(2)*sigma))
                                       - erf((z-center)/(np.sqrt(2)*sigma)))
    return radial[:, None]*axial[None, :]


class DiffusionTests(unittest.TestCase):
    def config(self, **kwargs):
        return replace(SimulationConfig(nr=8, nz=12, delta=1e-4, electrode_gap=6e-4,
            z0=4e-4, voltage=0., nmax=0., background_ionization=0.,
            electron_diffusion=.5, max_step=0), **kwargs)

    def step(self, state, dt, coefficient=None):
        if coefficient is not None:
            state.de[:] = coefficient
        if getattr(self, '_transport_state', None) is not state:
            self._transport_state = state
            self._transport = ElectronTransport(state, state.dr[0])
        return self._transport.step(state, dt)

    def test_config_validation_and_coefficient_field(self):
        for coefficient in (-1., np.nan, np.inf, True, None, '1'):
            with self.subTest(coefficient=coefficient), self.assertRaises(ValueError):
                initialize_simulation(self.config(electron_diffusion=coefficient))
        state = initialize_simulation(self.config())
        assert_array_equal(state.de[state.gas_mask], .5)
        assert_array_equal(state.de[~state.gas_mask], 0.)
        # 固定モードも輸送CFLを使うため、共通の初期化で検査する。
        for cfl in (0., 1.1, np.nan, True, None):
            with self.subTest(cfl=cfl), self.assertRaises(ValueError):
                initialize_simulation(self.config(dt_cfl_number=cfl))

    def test_harmonic_face_and_half_cell_absorption(self):
        grid = initialize_grid(SimulationConfig(nr=2, nz=2, delta=1., electrode_gap=10.))
        coefficient = np.array([[1., 9.], [1., 9.]])
        gas = np.ones((2, 2), dtype=bool)
        radial, axial = grid['grid'].split_faces(grid['grid'].diffusion_rates(coefficient, gas))
        assert_array_equal(radial[0], 0.)
        assert_allclose(axial[:, 1]/grid['sz'][:, 0], 1.8)
        assert_allclose(axial[:, 0]/grid['sz'][:, 0], 2.)
        assert_allclose(axial[:, -1]/grid['sz'][:, 0], 18.)
        gas[:, 1] = False
        _, axial = grid['grid'].split_faces(grid['grid'].diffusion_rates(coefficient, gas))
        assert_allclose(axial[:, 1]/grid['sz'][:, 0], 2.)
        assert_array_equal(axial[:, -1], 0.)
        # D=0の気体層は、調和平均では拡散障壁となる。
        coefficient[:, 1] = 0.
        _, axial = grid['grid'].split_faces(grid['grid'].diffusion_rates(coefficient, np.ones_like(gas)))
        assert_array_equal(axial[:, 1], 0.)

    def test_zero_coefficient_and_zero_velocity_keep_zero_state(self):
        state = initialize_simulation(self.config())
        state.de[:] = 0.
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            losses = self.step(state, 1e-8)
        assert_array_equal(state.ne, 0.)
        assert_array_equal(losses, 0.)

    def test_diffusion_limit_excludes_electrode_receiving_flux(self):
        config = self.config(nr=3, nz=3, delta=1., electrode_gap=1., cathode_radius=.3)
        grid = initialize_grid(config)
        coefficient = np.zeros((3, 3))
        coefficient[1, 1] = 1.
        state = SimpleNamespace(**grid, de=coefficient)
        # この気体セルはr下側の電極へだけ拡散する。他の気体はD=0。
        self.assertFalse(grid['gas_mask'][0, 1])
        self.assertTrue(grid['gas_mask'][1, 1])
        # 気体の体積1.5 / (D*面積1/半セル幅0.5) = 0.75。
        # 電極側の小さい体積0.5を制約の評価へ混ぜない。
        self.assertAlmostEqual(diffusion_limit(state, config), .75)

    def test_diffusion_does_not_emit_from_empty_electrode_or_boundaries(self):
        state = initialize_simulation(self.config())
        losses = self.step(state, diffusion_limit(state, self.config())*.5)
        assert_array_equal(state.ne, 0.)
        assert_array_equal(losses, 0.)

    def test_mixed_transport_preserves_particles_and_nonnegativity(self):
        config = self.config()
        state = initialize_simulation(config)
        rng = np.random.default_rng(31)
        state.ne[:] = rng.random(state.ne.shape)*state.gas_mask
        state.de[:] = (.1+rng.random(state.ne.shape))*state.gas_mask
        state.vex[:] = 500.
        state.vey[:] = -700.
        initial = np.sum(state.ne*state.volume)*2*np.pi
        loss = np.zeros(3)
        dt = transport_limit(state, config)*.5
        for _ in range(100):
            loss += self.step(state, dt)
        final = np.sum(state.ne*state.volume)*2*np.pi
        self.assertAlmostEqual((final+loss.sum())/initial, 1., places=13)
        self.assertTrue(np.all(state.ne >= 0.))
        self.assertTrue(np.all(loss >= 0.))
        assert_array_equal(state.ne[~state.gas_mask], 0.)

    def test_combined_limit_is_needed_for_simultaneous_update(self):
        config = self.config(nr=3, nz=3, delta=1., electrode_gap=10., electron_diffusion=1.)
        state = initialize_simulation(config)
        state.vex[:] = state.vey[:] = 3.
        state.de[:] = 0.
        drift_bound = transport_limit(state, config)
        state.de[:] = 1.
        separate_bound = min(drift_bound, diffusion_limit(state, config))
        state.ne[-1, -1] = 1.
        initial = state.ne.copy()
        self.step(state, separate_bound)
        self.assertLess(state.ne.min(), 0.)  # 個別上限の最小だけでは不十分。
        state.ne[:] = initial
        limits = timestep_limits(state, config)
        self.assertLess(limits.dt, separate_bound)
        self.step(state, limits.dt)
        self.assertTrue(np.all(state.ne >= 0.))

    def test_fixed_mode_rejects_unsafe_dt_and_adaptive_mode_shrinks_it(self):
        config = self.config(dt=1e-6, dt_max=1e-6, electron_diffusion=1.,
                             adaptive_timestep=False, t_end=1e-8)
        with self.assertRaisesRegex(TimestepError, 'drift-diffusion CFL'):
            run_simulation(config, write_output=False, verbose=False)
        state = run_simulation(replace(config, adaptive_timestep=True), write_output=False, verbose=False)
        assert_array_equal(state.ne, 0.)

    def test_rejected_diffusion_trial_rolls_back_density_and_losses(self):
        config = self.config()
        state = initialize_simulation(config)
        state.ne[:] = 1e15*state.gas_mask
        initial = np.sum(state.ne*state.volume)*2*np.pi
        calls = []

        real_step = ElectronTransport.step

        def fail_first(operator, current, dt):
            losses = real_step(operator, current, dt)
            calls.append(1)
            if len(calls) == 1:
                state.ne[state.gas_mask] = -1.
            return losses

        dt = transport_limit(state, config)*.1
        with patch.object(ElectronTransport, 'step', autospec=True, side_effect=fail_first):
            accepted, retries = try_density_step(state, config, dt)
        self.assertEqual(retries, 1)
        self.assertEqual(accepted, dt/2)
        final = np.sum(state.ne*state.volume)*2*np.pi
        self.assertAlmostEqual((final+state.electron_losses.sum())/initial, 1., places=13)

    def test_gaussian_spreading_converges_with_grid_refinement(self):
        errors = []
        sigma, coefficient, end, center = 1.5e-4, 1., 5e-9, 1e-3
        for spacing in (1e-4, 5e-5, 2.5e-5):
            config = self.config(nr=round(1e-3/spacing), nz=round(2e-3/spacing),
                                 delta=spacing, electrode_gap=3e-3, electron_diffusion=coefficient)
            grid = initialize_grid(config)
            density = gaussian_average(grid, sigma, center)
            state = SimpleNamespace(**grid, ne=density, de=np.full_like(density, coefficient),
                                    vex=np.zeros((config.nr+1, config.nz)),
                                    vey=np.zeros((config.nr, config.nz+1)))
            transport = ElectronTransport(state, spacing)
            count = math.ceil(end/(.05*diffusion_limit(state, config)))
            for _ in range(count):
                transport.step(state, end/count)
            spread = np.sqrt(sigma*sigma+2*coefficient*end)
            exact = (sigma/spread)**3*gaussian_average(grid, spread, center)
            errors.append(np.sum(np.abs(density-exact)*grid['volume'])/np.sum(exact*grid['volume']))
        self.assertLess(errors[1], errors[0]/2)
        self.assertLess(errors[2], errors[1]/2)
        self.assertLess(errors[-1], .01)

    def test_cli_and_output_record_diffusion_condition(self):
        with patch('streamer_drift_diffusion.__main__.run_simulation') as run:
            main(['--electron-diffusion', '.25', '--no-output'])
            self.assertEqual(run.call_args.args[0].electron_diffusion, .25)
        config = self.config(adaptive_timestep=True, t_end=2e-12, output_interval=1e-12)
        with tempfile.TemporaryDirectory() as directory:
            run_simulation(config, output_dir=directory, verbose=False)
            metadata = json.loads((Path(directory)/'run_config.json').read_text())
            self.assertEqual(metadata['config']['electron_diffusion'], .5)
            self.assertEqual(metadata['transport_backend'], 'fipy')
            self.assertEqual(metadata['transport_time_scheme'], 'explicit Euler')
            self.assertIn('fipy_version', metadata)
            self.assertIn('fipy_solver_suite', metadata)
            with (Path(directory)/'timestep_history.csv').open() as history:
                rows = list(csv.DictReader(history))
            self.assertTrue(rows)
            self.assertTrue(all(float(row['dt_diffusion_s']) > 0. for row in rows))
            self.assertTrue(all(float(row['dt_s']) <= float(row['dt_diffusion_s']) for row in rows))

    def test_forward_euler_time_convergence_against_discrete_eigenmode(self):
        # 2円環の半セル吸収を含む固有モードで、空間誤差を除く。
        size, coefficient, end = 8, .2, 1.
        radial = coefficient*np.array([[-2., 2.], [2/3, -10/3]])
        eigenvalues, eigenvectors = np.linalg.eig(radial)
        assert_allclose(eigenvalues.imag, 0., atol=1e-15)
        eigenvalues, eigenvectors = eigenvalues.real, eigenvectors.real
        slowest = np.argmax(eigenvalues)
        mode = eigenvectors[:, slowest]
        mode *= np.sign(mode[0])
        initial = mode[:, None]*np.sin(np.pi*(np.arange(size)+.5)/size)
        exact = initial*np.exp((eigenvalues[slowest]
                               -4*coefficient*np.sin(np.pi/(2*size))**2)*end)
        errors = []
        for count in (10, 20, 40):
            config = self.config(nr=2, nz=size, delta=1., electrode_gap=100.,
                                 electron_diffusion=coefficient)
            grid = initialize_grid(config)
            state = SimpleNamespace(**grid, ne=initial.copy(),
                                    de=np.full_like(initial, coefficient),
                                    vex=np.zeros((3, size)), vey=np.zeros((2, size+1)))
            transport = ElectronTransport(state, 1.)
            for _ in range(count):
                transport.step(state, end/count)
            errors.append(float(np.max(np.abs(state.ne-exact))))
        self.assertTrue(1.9 < errors[0]/errors[1] < 2.1)
        self.assertTrue(1.9 < errors[1]/errors[2] < 2.1)


if __name__ == '__main__':
    unittest.main()
