"""Helmholtz光電離の空間応答、粒子収支、フレーム出力を検証する。"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from streamer_drift_diffusion import SimulationConfig, initialize_simulation, run_simulation
from streamer_drift_diffusion.grid import CylindricalGrid
from streamer_drift_diffusion.photoionization import PhotoionizationSolver
from streamer_drift_diffusion.reactions import apply_reactions
from streamer_drift_diffusion.timestepping import chemistry_limit


class PhotoionizationTests(unittest.TestCase):
    """独立な積分核と局所粒子収支を基準にする。"""

    def test_point_source_matches_far_field_kernel_and_reuses_factors(self):
        """開放境界から離れた軸上で3項Green関数と比較する。"""
        config = SimulationConfig(nr=80, nz=160, delta=1e-4, photoionization=True)
        grid = CylindricalGrid(config)
        solver = PhotoionizationSolver(grid, np.ones(grid.shape, dtype=bool), config)
        impact = np.zeros(grid.shape)
        center = config.nz // 2
        impact[0, center] = 1e28
        rate = solver.solve(impact).copy()
        positions = center + np.array([8, 12, 20])
        distance = grid.zh[positions] - grid.zh[center]
        # 元論文の cm・Torr 単位の核を、軸上点源の積分発光量に掛ける。
        pressure = 760 * config.oxygen_fraction
        photons = (impact[0, center] * 2*np.pi*grid.volume[0, center]
                   * 0.075 * 40/(1013.25+40))
        kernel = sum(a*(100*pressure)**2 * np.exp(-b*pressure*100*distance)/(4*np.pi*distance)
                     for a, b in ((1.986e-4, 0.0553), (0.0051, 0.1460), (0.4886, 0.89)))
        assert_allclose(rate[0, positions], photons * kernel, rtol=0.06)
        factors = solver._factors
        assert_allclose(solver.solve(2*impact), 2*rate, rtol=1e-14)
        self.assertIs(solver._factors, factors)
        assert_array_equal(solver.solve(0*impact), 0.)

    def test_nonlocal_source_conserves_oxygen_and_charge(self):
        """電子衝突項をゼロにして光電離だけの粒子更新を確認する。"""
        config = SimulationConfig(nr=8, nz=12, delta=1e-4, electrode_gap=5.5e-4,
                                  cathode_radius=2.5e-4, nmax=0., background_ionization=0.,
                                  photoionization=True)
        state = initialize_simulation(config)
        impact = np.zeros(state.grid.shape)
        impact[0, 2] = 1e35
        solver = state.photoionization
        assert solver is not None
        rate, components = solver.solve(impact, return_components=True)
        self.assertGreater(rate[0, 1], 0.)
        self.assertTrue((~state.gas_mask).any())
        assert_array_equal(rate[~state.gas_mask], 0.)
        assert_array_equal(components[:, ~state.gas_mask], 0.)
        assert_allclose(components.sum(axis=0), rate, rtol=1e-14)
        dt = chemistry_limit(state, config) / 2
        self.assertTrue(np.isfinite(dt) and dt > 0)
        oxygen = state.o2.copy()
        apply_reactions(state, config, dt=dt)
        assert_array_equal(state.ne, dt * rate)
        assert_array_equal(state.o2p, state.ne)
        assert_allclose(state.o2 + state.o2p, oxygen, rtol=1e-15)

    def test_simulation_saves_photoionization_at_final_time(self):
        """通常の適応刻みループで源を作り、同時刻のCSVに書く。"""
        config = SimulationConfig(nr=8, nz=12, delta=1e-4, electrode_gap=5.5e-4,
                                  cathode_radius=2.5e-4, z0=4e-4, voltage=500.,
                                  photoionization=True, adaptive_timestep=True, t_end=3e-12)
        with TemporaryDirectory() as folder:
            state = run_simulation(config, output_dir=folder, verbose=False)
            solver = state.photoionization
            assert solver is not None
            self.assertGreater(float(solver.rate.max()), 0.)
            self.assertEqual(solver._factors, ())
            output = Path(folder)
            saved = np.loadtxt(output / "Sph1.csv").reshape(state.grid.shape)
            assert_allclose(saved, solver.rate, rtol=5e-7)
            metadata = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["photoionization_model"], "bourdon-2007-helmholtz-3")
