"""適応刻みの保存則、境界流出率、再試行、実時刻出力を検証する。"""
import csv
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from streamer_drift_diffusion import SimulationConfig, initialize_simulation, run_simulation, poisson_calc
from streamer_drift_diffusion.simulation import prepare_step, _same_time
from streamer_drift_diffusion.transport import ElectronTransport
from streamer_drift_diffusion.reactions import apply_reactions
from streamer_drift_diffusion.output import load_output_times
from streamer_drift_diffusion.transport_coefficients import electron_mobility
from streamer_drift_diffusion.timestepping import (
    DENSITY_NAMES, TimestepError, chemistry_limit, dielectric_limit,
    timestep_limits, transport_limit, try_density_step,
)


class AdaptiveTests(unittest.TestCase):
    def config(self, **kwargs):
        base = SimulationConfig(nr=6, nz=8, z0=50e-6, cathode_radius=30e-6,
                                adaptive_timestep=True, t_end=3.5e-12,
                                output_interval=1.3e-12, voltage=0., nmax=0.)
        return replace(base, **kwargs)

    def test_transport_bound_uses_all_outgoing_faces(self):
        config = self.config()
        state = initialize_simulation(config)
        volume = state.volume[1, 1]
        state.vex[1, 1] = -2.*volume/state.sr[1, 1]
        state.vex[2, 1] = 3.*volume/state.sr[2, 1]
        state.vey[1, 1] = -4.*volume/state.sz[1, 1]
        state.vey[1, 2] = 5.*volume/state.sz[1, 1]
        dt = transport_limit(state, config)
        self.assertAlmostEqual(dt, config.dt_cfl_number / 14.)
        transport = ElectronTransport(state, config.delta)
        for seed in range(5):
            state.ne[:] = np.random.default_rng(seed).random(state.ne.shape)
            transport.step(state, dt)
            self.assertTrue(np.all(state.ne >= 0))

    def test_dielectric_zero_field_is_not_zero_conductivity(self):
        config = self.config()
        state = initialize_simulation(config)
        state.abs_e[:] = 100.
        mobility = float(electron_mobility(np.array(100.), config))
        state.mobility_r[:] = state.mobility_z[:] = mobility
        state.ne[:] = 1e20
        uniform = dielectric_limit(state, config)
        state.abs_e[2, 3] = 0.
        state.mobility_r[2:4, 3] = 0.
        state.mobility_z[2, 3:5] = 0.
        state.ne[2, 3] = 3e20
        self.assertAlmostEqual(dielectric_limit(state, config) / uniform, 1 / 3)
        state.abs_e[:] = 0.
        state.mobility_r[:] = state.mobility_z[:] = 0.
        self.assertEqual(dielectric_limit(state, config), np.inf)
        state.abs_e[:] = 1e-40
        state.mobility_r[:] = state.mobility_z[:] = float(electron_mobility(np.array(1e-40), config))
        self.assertLess(dielectric_limit(state, config), uniform)

    def test_chemistry_limits_growth_and_losses_without_cancellation(self):
        config = self.config()
        state = initialize_simulation(config)
        state.n2[:] = 2.
        state.o2[:] = 3.
        state.ne[:] = 1.
        state.k1[:] = 4.
        state.k2[:] = 5.
        state.k3[:] = 6.
        self.assertAlmostEqual(chemistry_limit(state, config), config.dt_chemistry_fraction / 89.)

    def test_background_rate_is_independent_of_dt_partition(self):
        config = self.config()
        whole, split = initialize_simulation(config), initialize_simulation(config)
        apply_reactions(whole, config, dt=config.dt)
        for _ in range(4):
            apply_reactions(split, config, dt=config.dt / 4)
        for name in DENSITY_NAMES:
            assert_array_equal(getattr(whole, name), getattr(split, name))
        assert_allclose(split.ne, split.n2p + split.o2p, rtol=1e-15)

    def test_rejected_trial_restores_every_density_before_retry(self):
        config = self.config()
        state = initialize_simulation(config)
        calls = []

        def fail_first(operator, current, dt):
            calls.append(dt)
            if len(calls) == 1:
                state.ne[:] = -1.
                state.n2p[:] = -10.
            return np.zeros(3)

        with patch.object(ElectronTransport, 'step', autospec=True, side_effect=fail_first):
            dt, retries = try_density_step(state, config, config.dt)
        self.assertEqual(retries, 1)
        self.assertEqual(dt, config.dt / 2)
        assert_array_equal(state.ne, np.full_like(state.ne, config.background_ionization / 2))
        assert_allclose(state.ne, state.n2p + state.o2p, rtol=1e-15)

    def test_failed_trial_leaves_original_state(self):
        config = self.config(max_step_retries=0)
        state = initialize_simulation(config)
        original = {name: getattr(state, name).copy() for name in DENSITY_NAMES}

        def fail(operator, current, dt):
            state.ne[:] = -1.
            state.o2[:] = 0.

        with patch.object(ElectronTransport, 'step', autospec=True, side_effect=fail):
            with self.assertRaises(TimestepError):
                try_density_step(state, config, config.dt)
        for name in DENSITY_NAMES:
            assert_array_equal(getattr(state, name), original[name])

    def test_small_density_growth_does_not_spuriously_halve_every_step(self):
        config = self.config()
        state = initialize_simulation(config)
        state.abs_e[:] = 100.
        state.mobility_r[:] = state.mobility_z[:] = float(electron_mobility(np.array(100.), config))
        state.ne[:] = 1e21
        state.n2p[:] = state.ne
        state.k1[:] = 1e-18
        dt = config.dt_safety_factor * dielectric_limit(state, config)
        actual, retries = try_density_step(state, config, dt)
        self.assertEqual(retries, 0)
        self.assertEqual(actual, dt)

    def test_output_times_background_and_initial_final_frames(self):
        config = self.config(dt_max=2e-12, dt_min=0.4e-12)
        with tempfile.TemporaryDirectory() as directory:
            state = run_simulation(config, output_dir=directory, verbose=False)
            with (Path(directory) / 'times.csv').open() as times_stream:
                rows = list(csv.DictReader(times_stream))
            times = [float(row['time_s']) for row in rows]
            assert_allclose(times, [0., 1.3e-12, 2.6e-12, 3.5e-12], rtol=1e-15, atol=0.)
            for row in rows:
                self.assertEqual(row['time_s'], row['field_time_s'])
            assert_array_equal(np.loadtxt(Path(directory) / 'ne0.csv'), np.zeros(config.nr * config.nz))
            assert_allclose(state.ne, config.background_ionization * config.t_end / config.dt, rtol=1e-15)
            assert_allclose(state.ne, state.n2p + state.o2p, rtol=1e-15)
            self.assertTrue((Path(directory) / 'run_config.json').is_file())
            self.assertTrue((Path(directory) / 'timestep_history.csv').is_file())
            with self.assertRaises(FileExistsError):
                run_simulation(config, output_dir=directory, verbose=False)

    def test_final_event_may_be_shorter_than_dt_min(self):
        config = self.config(t_end=1.1e-12, dt_min=0.4e-12, output_interval=2e-12)
        state = run_simulation(config, write_output=False, verbose=False)
        assert_allclose(state.ne, config.background_ionization * 1.1, rtol=1e-15)

    def test_unstable_limit_is_not_rounded_up_to_dt_min(self):
        config = self.config(voltage=4000., nmax=1e25, dt_min=1e-12)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(TimestepError):
                run_simulation(config, output_dir=directory, verbose=False)
            self.assertTrue((Path(directory) / 'failure.json').exists())
            self.assertTrue((Path(directory) / 'failure_state.npz').exists())

    def test_zero_end_time_and_disabled_output(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / 'missing'
            state = run_simulation(self.config(t_end=0.), output_dir=missing, write_output=False, verbose=False)
            self.assertFalse(missing.exists())
            assert_array_equal(state.ne, np.zeros_like(state.ne))

    def test_plot_times_prefer_metadata_and_reject_missing_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            assert_allclose(load_output_times(directory, [0, 100], 1e-12), [0., 1e-10])
            path = Path(directory) / 'times.csv'
            path.write_text('frame,time_s\n0,0\n1,3e-12\n', encoding='utf-8')
            assert_allclose(load_output_times(directory, [0, 1], 1.), [0., 3e-12])
            with self.assertRaises(ValueError):
                load_output_times(directory, [2])
            path.write_text('frame,time_s\n0,0\n0,1\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                load_output_times(directory, [0])

    def test_accepted_step_guard_stops_instead_of_returning_early(self):
        with self.assertRaises(TimestepError):
            run_simulation(self.config(adaptive_max_steps=1), write_output=False, verbose=False)

    def test_dense_plasma_shrinks_dt_and_obeys_all_limits(self):
        config = self.config(voltage=500., nmax=1e22, t_end=1e-12)
        with tempfile.TemporaryDirectory() as directory:
            state = run_simulation(config, output_dir=directory, verbose=False)
            with (Path(directory) / 'timestep_history.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertGreater(len(rows), 1)
            for row in rows:
                bound = config.dt_safety_factor * min(float(row[key]) for key in
                    ('dt_cfl_s', 'dt_dielectric_s', 'dt_chemistry_s'))
                self.assertLessEqual(float(row['dt_s']), bound * (1 + 1e-12))
            self.assertTrue(np.all(np.isfinite(state.ne)))
            self.assertTrue(np.all(state.ne >= 0))

    def test_invalid_adaptive_parameters(self):
        for kwargs in ({'t_end': None}, {'t_end': np.nan}, {'dt_max': 0.},
                       {'dt_min': 2e-12}, {'dt_safety_factor': 0.}, {'dt_cfl_number': 1.1},
                       {'dt_chemistry_fraction': 0.}, {'output_interval': 0.},
                       {'dt_growth_factor': .9}, {'max_step_retries': -1},
                       {'adaptive_max_steps': True}, {'poisson_max_iterations': 0},
                       {'nitrogen_fraction': .7}, {'background_ionization': -1.}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                initialize_simulation(self.config(**kwargs))

    def test_field_is_recomputed_at_the_density_output_time(self):
        config = self.config(voltage=500., nmax=1e17, t_end=2e-12)
        state = run_simulation(config, write_output=False, verbose=False)
        phi = state.phi.copy()
        prepare_step(state, config, config.t_end, verbose=False)
        assert_allclose(state.phi, phi, rtol=1e-6, atol=1e-7)
        self.assertTrue(np.all(state.ne >= 0))

    def test_waveform_knots_are_not_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'short_pulse.txt'
            path.write_text('0 0\n0.001 1\n0.002 0\n', encoding='utf-8')
            config = self.config(t_end=3e-12, dt_max=1e-9, voltage_mode='impulse',
                                 voltage_waveform_file=str(path), background_ionization=0.)
            output = Path(directory) / 'output'
            run_simulation(config, output_dir=output, verbose=False)
            with (output / 'timestep_history.csv').open() as history:
                rows = list(csv.DictReader(history))
            times = np.array([float(row['time_s']) for row in rows])
            self.assertTrue(np.any(np.isclose(times, 1e-12, rtol=1e-14, atol=0.)))
            self.assertTrue(np.any(np.isclose(times, 2e-12, rtol=1e-14, atol=0.)))

    def test_same_time_uses_local_float_spacing(self):
        time = 0.7e-9
        self.assertTrue(_same_time(time, np.nextafter(time, np.inf)))
        self.assertFalse(_same_time(time, time + 16 * np.spacing(time)))
        self.assertFalse(_same_time(0., 1e-30))
        self.assertTrue(_same_time(0., 0.))
        self.assertFalse(_same_time(time, np.inf))
        self.assertFalse(_same_time(np.nan, time))

    def test_coincident_waveform_output_events_do_not_add_microsteps(self):
        # 同梱波形は0.05 ns間隔。倍率0.25で12.5 psとなり、
        # 1 ps*12 + 0.5 psの13更新を56区間、合計728更新で進む。
        # 電圧と背景源を0にし、時刻制御だけを検証する。
        config = self.config(voltage_mode='impulse', impulse_peak_voltage=0.,
                             impulse_time_scale=0.25, background_ionization=0.,
                             t_end=0.7e-9, output_interval=0.1e-9)
        with tempfile.TemporaryDirectory() as directory:
            run_simulation(config, output_dir=directory, verbose=False)
            with (Path(directory) / 'timestep_history.csv').open() as stream:
                history = list(csv.DictReader(stream))
            self.assertEqual(len(history), 728)
            dts = np.array([float(row['dt_s']) for row in history])
            self.assertGreater(float(dts.min()), 0.49e-12)
            self.assertLessEqual(float(dts.max()), config.dt)
            with (Path(directory) / 'times.csv').open() as stream:
                frames = list(csv.DictReader(stream))
            self.assertEqual([int(row['step']) for row in frames], [104 * i for i in range(8)])
            times = np.array([float(row['time_s']) for row in frames])
            assert_allclose(times, np.arange(8) * config.output_interval, rtol=1e-15, atol=0.)
            self.assertTrue(np.all(np.diff(times) > 0))
            self.assertEqual(times[-1], config.t_end)
            self.assertEqual(len(list(Path(directory).glob('ne*.csv'))), 8)

    def test_output_and_end_one_ulp_apart_are_saved_once(self):
        config = self.config(t_end=float(np.nextafter(2e-12, np.inf)), output_interval=2e-12)
        with tempfile.TemporaryDirectory() as directory:
            state = run_simulation(config, output_dir=directory, verbose=False)
            with (Path(directory) / 'times.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual(int(rows[-1]['step']), 2)
            self.assertEqual(float(rows[-1]['time_s']), config.t_end)
            self.assertEqual(rows[-1]['time_s'], rows[-1]['field_time_s'])
            self.assertLessEqual(float(rows[-1]['dt_s']), config.dt)
            assert_allclose(state.ne, config.background_ionization * config.t_end / config.dt, rtol=1e-15)

    def test_resolvable_end_fraction_below_dt_min_is_preserved(self):
        config = self.config(t_end=1e-12 + 1e-20, output_interval=2e-12)
        with tempfile.TemporaryDirectory() as directory:
            run_simulation(config, output_dir=directory, verbose=False)
            with (Path(directory) / 'timestep_history.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual(float(rows[-1]['time_s']), config.t_end)
            self.assertGreater(float(rows[-1]['dt_s']), 0.)
            self.assertLess(float(rows[-1]['dt_s']), config.dt_min)

    def test_adaptive_dc_polarities_and_impulse(self):
        for changes in ({'voltage': 500.}, {'voltage': -500.},
                        {'voltage_mode': 'impulse', 'impulse_peak_voltage': 500.}):
            with self.subTest(changes=changes):
                state = run_simulation(self.config(nmax=1e17, t_end=20e-12, **changes),
                                       write_output=False, verbose=False)
                for name in DENSITY_NAMES:
                    self.assertTrue(np.all(np.isfinite(getattr(state, name))))
                    self.assertTrue(np.all(getattr(state, name) >= 0))
                self.assertTrue(np.all(np.isfinite(state.abs_e)))

    def test_poisson_iteration_cap(self):
        state = initialize_simulation(self.config())
        with self.assertRaises(RuntimeError):
            poisson_calc(state.pphi, np.ones_like(state.ne), state.p1, state.p2, state.p3,
                         state.p4, state.p5, max_iterations=1, tolerance=1e-15, verbose=False)


if __name__ == '__main__':
    unittest.main()
