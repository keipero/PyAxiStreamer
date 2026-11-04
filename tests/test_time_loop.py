"""固定／適応の共通時計・出力と、固定刻みの停止方針を検証する。"""
import contextlib
import csv
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from streamer_drift_diffusion import SimulationConfig, initialize_simulation, run_simulation
from streamer_drift_diffusion.__main__ import main
from streamer_drift_diffusion.simulation import prepare_step
from streamer_drift_diffusion.timestepping import advance_density, DENSITY_NAMES, TimestepError
from streamer_drift_diffusion.transport import ElectronTransport


def rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


class TimeLoopTests(unittest.TestCase):
    def config(self, **changes):
        return replace(SimulationConfig(nr=6, nz=8, z0=50e-6, voltage=0., nmax=0.,
            dt=1e-12, t_end=3.5e-12, output_interval=1.3e-12), **changes)

    def test_modes_share_frame_times_initial_state_and_background_rate(self):
        answers = []
        with tempfile.TemporaryDirectory() as temporary:
            for adaptive in (False, True):
                config = self.config(adaptive_timestep=adaptive)
                folder = Path(temporary)/str(adaptive)
                state = run_simulation(config, output_dir=folder, verbose=False)
                frames = rows(folder/'times.csv')
                assert_allclose([float(r['time_s']) for r in frames],
                                [0., 1.3e-12, 2.6e-12, 3.5e-12], rtol=1e-15, atol=0.)
                for frame in frames:
                    self.assertEqual(frame['time_s'], frame['field_time_s'])
                    density = np.loadtxt(folder/f"ne{frame['frame']}.csv")
                    expected = config.background_ionization*float(frame['time_s'])/config.dt
                    assert_allclose(density, expected, rtol=1e-6, atol=0.)
                assert_array_equal(np.loadtxt(folder/'ne0.csv'), 0.)
                assert_allclose(state.ne, config.background_ionization*3.5, rtol=1e-15)
                answers.append(state.ne.copy())
                with self.assertRaises(FileExistsError):
                    run_simulation(config, output_dir=folder, verbose=False)
            assert_array_equal(*answers)

    def test_fixed_steps_shorten_only_at_events_and_do_not_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            run_simulation(self.config(), output_dir=folder, verbose=False)
            history = rows(folder/'timestep_history.csv')
            self.assertTrue(any(float(r['dt_s']) < 1e-12*(1-1e-12) for r in history))
            for row in history:
                self.assertEqual(int(row['retries']), 0)
                self.assertLessEqual(float(row['dt_s']), 1e-12*(1+1e-12))
                self.assertIn(row['limiting'], ('fixed', 'output', 't_end'))
                if float(row['dt_s']) < 1e-12*(1-1e-12):
                    self.assertIn(row['limiting'], ('output', 't_end'))

    def test_fixed_failure_restores_all_densities_without_halving(self):
        config = self.config()
        state = initialize_simulation(config)
        before = {name: getattr(state, name).copy() for name in DENSITY_NAMES}

        def fail(operator, current, dt, *, prepared=False):
            current.ne[:] = -1.
            current.o2[:] = 0.
            return np.ones(3)

        with patch.object(ElectronTransport, 'step', autospec=True, side_effect=fail) as stepped:
            with self.assertRaises(TimestepError):
                advance_density(state, config, config.dt)
            self.assertEqual(stepped.call_count, 1)
        for name in DENSITY_NAMES:
            assert_array_equal(getattr(state, name), before[name])
        assert_array_equal(state.electron_losses, 0.)

    def test_fixed_instability_stops_before_density_update_and_saves_failure(self):
        config = self.config(nmax=1e25, voltage=4000., t_end=2e-12, output_interval=1e-10)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            with patch.object(ElectronTransport, 'step', autospec=True) as stepped:
                with self.assertRaisesRegex(TimestepError, 'fixed dt exceeds'):
                    run_simulation(config, output_dir=folder, verbose=False)
                self.assertFalse(stepped.called)
            self.assertTrue((folder/'failure.json').exists())
            with np.load(folder/'failure_state.npz') as snapshot:
                assert_array_equal(snapshot['electron_losses'], 0.)
            self.assertEqual(len(rows(folder/'timestep_history.csv')), 0)

    def test_fixed_final_fields_are_at_end_time(self):
        config = self.config(nmax=1e17, voltage=500., t_end=2e-12)
        state = run_simulation(config, write_output=False, verbose=False)
        before = {name: getattr(state, name).copy() for name in ('phi', 'rho', 'ex', 'ey')}
        prepare_step(state, config, config.t_end, verbose=False)
        for name in before:
            assert_allclose(getattr(state, name), before[name], rtol=1e-13, atol=0.)

    def test_fixed_mode_resolves_waveform_knots(self):
        with tempfile.TemporaryDirectory() as temporary:
            waveform = Path(temporary)/'pulse.txt'
            waveform.write_text('0 0\n0.001 1\n0.002 0\n', encoding='utf-8')
            config = self.config(dt=3e-12, t_end=3e-12, output_interval=1e-11,
                voltage_mode='impulse', voltage_waveform_file=str(waveform), background_ionization=0.)
            folder = Path(temporary)/'output'
            run_simulation(config, output_dir=folder, verbose=False)
            history = rows(folder/'timestep_history.csv')
            assert_allclose([float(r['time_s']) for r in history], [1e-12, 2e-12, 3e-12], rtol=1e-14, atol=0.)
            self.assertEqual([r['limiting'] for r in history], ['waveform', 'waveform', 't_end'])

    def test_zero_duration_writes_only_initial_frame_in_both_modes(self):
        with tempfile.TemporaryDirectory() as temporary:
            for adaptive in (False, True):
                folder = Path(temporary)/str(adaptive)
                with patch.object(ElectronTransport, 'step', autospec=True) as stepped:
                    run_simulation(self.config(t_end=0., adaptive_timestep=adaptive),
                                   output_dir=folder, verbose=False)
                    self.assertFalse(stepped.called)
                self.assertEqual(len(rows(folder/'times.csv')), 1)
                self.assertEqual(rows(folder/'timestep_history.csv'), [])
                self.assertFalse((folder/'ne1.csv').exists())

    def test_common_cli_and_removed_step_options(self):
        with patch('streamer_drift_diffusion.__main__.run_simulation') as run:
            main(['--t-end', '2e-12', '--output-interval', '5e-13', '--max-steps', '10', '--no-output'])
            config = run.call_args.args[0]
            self.assertEqual((config.t_end, config.output_interval, config.max_steps), (2e-12, 5e-13, 10))
        for option in ('--steps', '--write-every', '--adaptive-max-steps'):
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main([option, '1'])


if __name__ == '__main__':
    unittest.main()
