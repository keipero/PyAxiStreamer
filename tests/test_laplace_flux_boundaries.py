"""上端・外周の法線電場を解析Laplace電場に固定する境界（空間電荷電位の同次Neumann）の検証。"""
from dataclasses import replace
import unittest

import numpy as np
from numpy.testing import assert_allclose

from streamer_drift_diffusion import SimulationConfig, hyperbolic_curve, initialize_simulation
from streamer_drift_diffusion.electrostatics import (calculate_electric_field, electrode_contact_radius,
                                                     laplace_gradient)
from streamer_drift_diffusion.simulation_types import molecular_density


class LaplaceFluxBoundaryTests(unittest.TestCase):
    def setUp(self):
        # 上端＝針先端の高さ（電極セルなし）のギャップだけの領域
        self.config = SimulationConfig(nr=40, nz=40, delta=25e-6, electrode_gap=1e-3,
                                       cathode_radius=1e-4, voltage=1000., nmax=0.,
                                       background_ionization=0., laplace_flux_boundaries="outer_and_top")
        self.state = initialize_simulation(self.config)
        self.assertTrue(self.state.gas_mask.all())

    def analytic(self):
        c = self.config
        return np.array([[hyperbolic_curve(c.voltage, c.cathode_radius, c.electrode_gap, r, z)
                          for z in self.state.zh] for r in self.state.rh])

    def test_charge_free_solution_matches_analytic_potential(self):
        potential = self.state.poisson.solve(np.zeros(self.state.ne.shape), self.config.voltage).copy()
        dirichlet = initialize_simulation(replace(self.config, laplace_flux_boundaries="none"))
        reference = dirichlet.poisson.solve(np.zeros(self.state.ne.shape), self.config.voltage)
        exact = self.analytic()
        error = np.abs(potential - exact).max() / self.config.voltage
        self.assertLess(error, 2e-2)
        # 同じ格子のDirichlet解と同程度の離散化誤差
        self.assertLess(error, 3 * np.abs(reference - exact).max() / self.config.voltage + 1e-3)

    def test_boundary_normal_field_is_fixed_to_laplace_field(self):
        state, config = self.state, self.config
        scale = molecular_density(config) * 1e-21
        state.poisson.solve(np.zeros(state.ne.shape), config.voltage)
        calculate_electric_field(state, config, voltage=config.voltage)
        free = state.rh > electrode_contact_radius(config)
        self.assertTrue(0 < np.count_nonzero(~free) < 5)  # 軸付近の数セルだけ電極電位に固定
        top = state.ez_face[:, -1].copy()
        outer = state.er_face[-1].copy()
        expected_top = [-config.voltage * laplace_gradient(config, r, config.nz * config.delta)[1] / scale
                        for r in state.rh[free]]
        assert_allclose(top[free], expected_top, rtol=1e-12)
        self.assertLess(top[0], 0.)  # 軸上の上端では電場は平板（-z）向き
        # 空間電荷を置いても、固定面の法線電場は変わらない（空間電荷電位の法線勾配0）
        rho = np.zeros(state.ne.shape)
        rho[:10, -5:] = 1e8
        state.poisson.solve(rho, config.voltage)
        calculate_electric_field(state, config, voltage=config.voltage)
        assert_allclose(state.ez_face[free, -1], top[free], rtol=1e-12)
        assert_allclose(state.er_face[-1], outer, rtol=1e-12)
        # 電極との接触面では電位を固定するので、正電荷で電場が強まる
        self.assertTrue(np.all(state.ez_face[~free, -1] > top[~free]))

    def test_contact_keeps_potential_near_electrode_voltage(self):
        # 上端付近の強い正電荷でも、電極接触があるため電位は印加電圧から浮かない
        state, config = self.state, self.config
        rho = np.zeros(state.ne.shape)
        rho[:10, -5:] = 1e10
        potential = state.poisson.solve(rho, config.voltage)
        self.assertLess(abs(potential[0, -1] - config.voltage), .05 * config.voltage)
        floating = initialize_simulation(replace(config, electrode_contact_radius=0.))
        lifted = floating.poisson.solve(rho, config.voltage)[0, -1]
        self.assertGreater(lifted - config.voltage, 3 * abs(potential[0, -1] - config.voltage))

    def test_gauss_law_with_fixed_boundary_flux(self):
        # 正電荷による電束の増分は、電位を固定した面（接地平面と電極接触面）だけから出ていく
        state, config = self.state, self.config
        grid = state.grid
        scale = molecular_density(config) * 1e-21
        bottom, top = grid.face_ids[1][:, 0], grid.face_ids[1][:, -1]
        contact = state.rh <= electrode_contact_radius(config)

        def outward_flux():
            calculate_electric_field(state, config, voltage=config.voltage)
            return (-(state.ez_face[:, 0] * grid.area[bottom]).sum()
                    + (state.ez_face[contact, -1] * grid.area[top][contact]).sum())

        state.poisson.solve(np.zeros(state.ne.shape), config.voltage)
        before = outward_flux()
        rho = np.zeros(state.ne.shape)
        rho[5:10, 10:20] = 1e8
        state.poisson.solve(rho, config.voltage)
        after = outward_flux()
        charge = (rho * grid.volume).sum()
        assert_allclose((after - before) * scale, charge, rtol=1e-10)

    def test_exclusive_with_flux_neumann(self):
        with self.assertRaises(ValueError):
            initialize_simulation(replace(self.config, flux_neumann_boundaries=True))
        with self.assertRaises(ValueError):
            initialize_simulation(replace(self.config, laplace_flux_boundaries="top"))

    def test_outer_only_keeps_dirichlet_top(self):
        # "outer"では外周だけ法線電場を固定し、上端は解析電位のDirichletのまま
        config = replace(self.config, laplace_flux_boundaries="outer")
        state = initialize_simulation(config)
        self.assertTrue(state.poisson.flux_faces[state.grid.face_ids[0][-1]].all())
        self.assertFalse(state.poisson.flux_faces[state.grid.face_ids[1][:, -1]].any())
        state.poisson.solve(np.zeros(state.ne.shape), config.voltage)
        calculate_electric_field(state, config, voltage=config.voltage)
        top, outer = state.ez_face[:, -1].copy(), state.er_face[-1].copy()
        rho = np.zeros(state.ne.shape)
        rho[20:35, 10:30] = 1e8
        state.poisson.solve(rho, config.voltage)
        calculate_electric_field(state, config, voltage=config.voltage)
        assert_allclose(state.er_face[-1], outer, rtol=1e-12)
        self.assertGreater(np.abs(state.ez_face[:, -1] - top).max(), 1e-3 * np.abs(top).max())


class ReactionBufferTests(unittest.TestCase):
    def test_buffer_cells_have_no_reactions_or_photoionization(self):
        config = SimulationConfig(nr=6, nz=10, delta=1e-4, electrode_gap=1e-3, cathode_radius=1e-4,
                                  voltage=0., nmax=0., background_ionization=1e12,
                                  reaction_buffer_height=3e-4)
        state = initialize_simulation(config)
        buffer = state.zh >= config.nz * config.delta - 3e-4
        self.assertEqual(np.count_nonzero(buffer), 3)
        self.assertFalse(state.reaction_mask[:, buffer].any())
        assert_allclose(state.reaction_mask[:, ~buffer], state.gas_mask[:, ~buffer])
        gas = state.gas_mask
        state.ne[gas], state.n2[gas], state.o2[gas] = 1e18, 2e25, 5e24
        state.abs_e[gas] = 300.
        from streamer_drift_diffusion.reactions import apply_reactions, calculate_reaction_coefficients
        calculate_reaction_coefficients(state)
        self.assertTrue(np.all(state.k1[:, buffer] == 0.))
        self.assertTrue(np.all(state.k1[state.reaction_mask] > 0.))
        state.photoionization.rate[:] = 1e25
        before = state.ne.copy()
        apply_reactions(state, config, dt=config.dt)
        assert_allclose(state.ne[:, buffer], before[:, buffer], rtol=0)
        self.assertTrue(np.all(state.ne[state.reaction_mask] > before[state.reaction_mask]))
        with self.assertRaises(ValueError):
            initialize_simulation(replace(config, reaction_buffer_height=-1.))


class UniformSeedTests(unittest.TestCase):
    def test_uniform_quasi_neutral_seed(self):
        config = SimulationConfig(nr=4, nz=6, delta=1e-4, electrode_gap=3e-4, cathode_radius=1e-4,
                                  voltage=0., nmax=0., background_ionization=0.,
                                  initial_uniform_density=5e9)
        state = initialize_simulation(config)
        gas = state.gas_mask
        self.assertFalse(gas.all())  # 電極セルを含む格子
        assert_allclose(state.ne[gas], 5e9, rtol=1e-14)
        self.assertTrue(np.all(state.ne[~gas] == 0.))
        assert_allclose(state.n2p[gas], config.nitrogen_fraction * 5e9, rtol=1e-14)
        assert_allclose(state.ne, state.n2p + state.o2p, rtol=1e-14)
        with self.assertRaises(ValueError):
            initialize_simulation(replace(config, initial_uniform_density=-1.))


if __name__ == "__main__":
    unittest.main()
