from dataclasses import fields, replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from streamer_drift_diffusion import (
    SimulationConfig, SimulationState, initialize_simulation, run_simulation,
    hyperbolic_curve, ElectronTransport, load_voltage_waveform, applied_voltage,
)
from streamer_drift_diffusion.output import write_c_matrix
from streamer_drift_diffusion.reactions import apply_reactions


class SimulationTests(unittest.TestCase):
    def small_config(self, **kwargs):
        return replace(SimulationConfig(nr=12, nz=10, t_end=3e-12,
                       z0=70e-6, cathode_radius=30e-6), **kwargs)

    def test_base_potential_boundaries(self):
        self.assertEqual(hyperbolic_curve(4000., 300e-6, 1e-3, 0., 0.), 0.)
        self.assertEqual(hyperbolic_curve(4000., 300e-6, 1e-3, 0., 1e-3), 4000.)

    def test_initial_neutrality_and_independent_arrays(self):
        config = self.small_config()
        state = initialize_simulation(config)
        self.assertEqual(state.ne.shape, (12, 10))
        self.assertEqual(state.rh[0], config.delta * .5)
        assert_array_equal(state.ne, state.n2p + state.o2p)
        matrices = [getattr(state, f.name) for f in fields(SimulationState)
                    if isinstance(getattr(state, f.name), np.ndarray) and getattr(state, f.name).ndim == 2]
        matrices += [state.ne, state.phi]  # FiPy正本のビューも物理配列の独立性検査へ含める。
        self.assertEqual(len(matrices), 27)
        for i, left in enumerate(matrices):
            for right in matrices[i + 1:]:
                self.assertFalse(np.shares_memory(left, right))



    def test_poisson_known_parabola_both_signs(self):
        config = SimulationConfig(nr=12, nz=16, delta=1., electrode_gap=32., poisson_tolerance=1e-12)
        state = initialize_simulation(config)
        height = config.nz * config.delta
        for sign in (1., -1.):
            expected = np.broadcast_to(sign*(4*state.zh*(height-state.zh)+config.delta**2)/height**2, (12, 16))
            state.radial_boundary[:] = expected[0]
            state.upper_boundary[:] = 0.
            answer = state.poisson.solve(np.full((12, 16), sign*8/height**2), 1.)
            assert_allclose(answer, expected, rtol=1e-12, atol=1e-12)


    def test_transport_top_boundary(self):
        config = self.small_config(nr=2, nz=2, delta=1., electrode_gap=100.)
        state = initialize_simulation(config)
        state.ne[:] = [[1., 2.], [3., 4.]]
        state.vey[:] = [1., 2., 2.]
        losses = ElectronTransport(state, config.delta).step(state, .1)
        # セル間面は片側勾配1とCFL 0.2で面密度を0.4補正する。
        assert_allclose(state.ne, [[.72, 1.88], [2.32, 3.88]])
        self.assertAlmostEqual(losses.sum(), 1.4 * 2*np.pi)

    def test_transport_outer_boundary(self):
        config = self.small_config(nr=2, nz=2, delta=1., electrode_gap=100.)
        state = initialize_simulation(config)
        state.ne[:] = [[1., 2.], [3., 4.]]
        state.vex[:] = [[0., 0.], [2., 2.], [2., 2.]]
        losses = ElectronTransport(state, config.delta).step(state, .1)
        # 軸側の勾配は1。内部面の補正は0.5 * (0.5 - 0.2)。
        assert_allclose(state.ne, [[.54, 1.14], [7/3 + .02, 3.22]])
        self.assertAlmostEqual(losses.sum(), 2.8 * 2*np.pi)

    def test_reaction_charge_conservation_and_background(self):
        config = self.small_config()
        state = initialize_simulation(config)
        state.k1[:] = 1e-17
        state.k2[:] = 1e-18
        state.k3[:] = 1e-43
        apply_reactions(state, config, dt=config.dt)
        assert_allclose(state.n2p + state.o2p, state.ne + state.om + state.o2m, rtol=1e-15)
        zero = run_simulation(self.small_config(voltage=0., nmax=0., t_end=3e-12),
                              write_output=False, verbose=False)
        assert_array_equal(zero.ne, np.full((12, 10), 3 * config.background_ionization))

    def test_bundled_impulse_and_scaling(self):
        config = SimulationConfig(voltage_mode="impulse")
        table = load_voltage_waveform(config.voltage_waveform_file)
        self.assertEqual(len(table.times), 20001)
        self.assertEqual(table.peak_voltage, 50000.)
        self.assertAlmostEqual(applied_voltage(config, 0., waveform=table), 244.3133)
        for scale in (.5, 1., 2.):
            scaled = replace(config, impulse_time_scale=scale)
            self.assertAlmostEqual(applied_voltage(scaled, 5.35e-9 * scale, waveform=table), 5000.)
            for time in (-1e-9, 0., .725e-9, 2e-9, 15e-9, 1000e-9):
                self.assertAlmostEqual(applied_voltage(scaled, time * scale, waveform=table),
                                       applied_voltage(config, time, waveform=table))

    def test_waveform_interpolation_endpoints_and_time_dependent_potential(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "triangle.txt"
            path.write_text("# ns kV\n0 0\n0.001 1\n0.002 0\n", encoding="utf-8")
            config = self.small_config(voltage_mode="impulse", impulse_peak_voltage=1000.,
                impulse_time_scale=2., voltage_waveform_file=str(path), t_end=2e-12,
                nmax=0., background_ionization=0.)
            table = load_voltage_waveform(path)
            self.assertEqual(applied_voltage(config, -1e-9, waveform=table), 0.)
            self.assertEqual(applied_voltage(config, 1e-9, waveform=table), 0.)
            self.assertAlmostEqual(applied_voltage(config, 1e-12, waveform=table), 500.)
            final = run_simulation(config, write_output=False, verbose=False)
            direct = run_simulation(replace(config, voltage_mode='dc', voltage=1000.), write_output=False, verbose=False)
            assert_allclose(final.phi, direct.phi, rtol=1e-14)
            assert_array_equal(final.ne, np.zeros_like(final.ne))

    def test_constant_impulse_equals_dc(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "constant.txt"
            path.write_text("0 1\n1 1\n", encoding="utf-8")
            dc = self.small_config(voltage=500.)
            pulse = replace(dc, voltage_mode="impulse", impulse_peak_voltage=500., voltage_waveform_file=str(path))
            left = run_simulation(dc, write_output=False, verbose=False)
            right = run_simulation(pulse, write_output=False, verbose=False)
            assert_array_equal(left.phi, right.phi)
            assert_array_equal(left.ne, right.ne)

    def test_invalid_waveforms(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.txt"
            for contents in ("0 1\n0 2\n", "1 1\n0 2\n", "0 NaN\n1 1\n", "0 1 2\n1 1\n",
                             "0 1\n", "time voltage\n1 1\n", "0 0\n1 0\n", "0 1\nInf 1\n"):
                path.write_text(contents, encoding="utf-8")
                with self.subTest(contents=contents), self.assertRaises(ValueError):
                    load_voltage_waveform(path)

    def test_invalid_config(self):
        for kwargs in ({"nr": 1}, {"nz": 1}, {"nr": 2.5}, {"output_interval": 0}, {"t_end": -1},
                       {"voltage_mode": "unknown"}, {"delta": 0}, {"dt": -1}, {"poisson_tolerance": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                initialize_simulation(self.small_config(**kwargs))
        for tolerance in (0., -.5, np.nan, np.inf):
            with self.assertRaises(ValueError):
                initialize_simulation(self.small_config(poisson_tolerance=tolerance))
        for scale in (0., -1., np.nan, np.inf):
            with self.assertRaises(ValueError):
                initialize_simulation(self.small_config(voltage_mode="impulse", impulse_time_scale=scale))
        with self.assertRaises(ValueError):
            applied_voltage(SimulationConfig(), np.nan)

    def test_csv_order_format_and_update_count(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            write_c_matrix(directory / "order.csv", np.array([[1., 2.], [3., 4.]]))
            self.assertEqual((directory / "order.csv").read_bytes(),
                             b"1.000000e+00\n2.000000e+00\n3.000000e+00\n4.000000e+00\n")
            (directory / "order.csv").unlink()
            final = run_simulation(self.small_config(output_interval=2e-12, t_end=4e-12), output_dir=directory, verbose=False)
            self.assertTrue((directory / "ne0.csv").exists())
            self.assertTrue((directory / "ne2.csv").exists())
            self.assertFalse((directory / "ne3.csv").exists())
            self.assertTrue(np.all(np.isfinite(final.ne)))
            self.assertTrue(np.all(final.ne >= 0))
            missing = directory / "not_created"
            run_simulation(self.small_config(t_end=1e-12), output_dir=missing, write_output=False, verbose=False)
            self.assertFalse(missing.exists())


if __name__ == "__main__":
    unittest.main()
