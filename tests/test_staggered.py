"""中心・面の配置、電場再構成、Poissonと輸送の整合性を検証する。"""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from streamer_drift_diffusion import SimulationConfig, initialize_simulation, run_simulation
from streamer_drift_diffusion.electrostatics import calculate_electric_field, discretization
from streamer_drift_diffusion.transport import ElectronTransport
from streamer_drift_diffusion.reactions import calculate_reaction_coefficients
from streamer_drift_diffusion.simulation import prepare_step
from streamer_drift_diffusion.simulation_types import molecular_density
from streamer_drift_diffusion.timestepping import TimestepError, dielectric_limit, transport_limit
from streamer_drift_diffusion.transport_coefficients import calculate_transport_coefficients, electron_drift


class StaggeredTests(unittest.TestCase):
    def config(self, **kwargs):
        return replace(SimulationConfig(nr=6, nz=8, delta=.1, electrode_gap=2.,
            voltage=1., nmax=0., background_ionization=0., max_step=0), **kwargs)

    def prescribe(self, state, config, potential):
        """既知の電位 [Td*m]と境界値を与え、場を計算する。"""
        scale = molecular_density(config)*1e-21
        state.phi[:] = scale*potential(state.rh[:, None], state.zh[None, :])
        state.radial_boundary[:] = scale*potential(config.nr*config.delta, state.zh)
        state.upper_boundary[:] = scale*potential(state.rh, config.nz*config.delta)
        calculate_electric_field(state, config, voltage=1.)

    def test_shapes_and_no_shared_storage(self):
        config = self.config()
        state = initialize_simulation(config)
        for name in ('phi', 'ne', 'ex', 'ey', 'abs_e', 'de'):
            self.assertEqual(getattr(state, name).shape, (6, 8))
        for name in ('er_face', 'vex', 'mobility_r'):
            self.assertEqual(getattr(state, name).shape, (7, 8))
        for name in ('ez_face', 'vey', 'mobility_z'):
            self.assertEqual(getattr(state, name).shape, (6, 9))
        self.assertFalse(np.shares_memory(state.ex, state.er_face))
        self.assertFalse(np.shares_memory(state.ey, state.ez_face))

    def test_uniform_axial_field_is_exact_including_boundaries(self):
        config = self.config()
        state = initialize_simulation(config)
        self.prescribe(state, config, lambda r, z: 70.*z + 0.*r)
        calculate_transport_coefficients(state, config)
        assert_allclose(state.er_face, 0., atol=1e-12)
        assert_allclose(state.ez_face, -70., rtol=1e-14)
        assert_allclose(state.ex, 0., atol=1e-12)
        assert_allclose(state.ey, -70., rtol=1e-14)
        assert_allclose(state.abs_e, 70., rtol=1e-14)
        assert_allclose(state.vex, 0., atol=1e-8)
        assert_allclose(state.vey, electron_drift(-70., 70.), rtol=1e-14)

    def test_polynomial_components_are_combined_at_cell_centers(self):
        config = self.config()
        state = initialize_simulation(config)
        self.prescribe(state, config, lambda r, z: r*r*z)
        r, z = state.rh[:, None], state.zh[None, :]
        # r=0を含む内部セルで、半径2次・軸1次の電位を正しく再構成できる。
        assert_allclose(state.ex[:-1], (-2*r*z)[:-1], atol=1e-14)
        assert_allclose(state.ey, np.broadcast_to(-r*r, state.ne.shape), atol=1e-14)
        assert_allclose(state.abs_e[:-1], np.hypot(2*r*z, r*r)[:-1], atol=1e-14)
        old_norm = np.hypot(state.er_face[:-1], state.ez_face[:, :-1])
        self.assertGreater(np.max(np.abs(old_norm[:-1]-state.abs_e[:-1])), .01)

    def test_face_mobility_uses_transverse_component_at_same_position(self):
        config = self.config()
        state = initialize_simulation(config)
        r, z = state.rh[:, None], state.zh[None, :]
        rf = np.arange(config.nr+1)[:, None]*config.delta
        zf = np.arange(config.nz+1)[None, :]*config.delta
        state.ex[:] = 30.+5*r+3*z
        state.ey[:] = 10.+3*r+2*z
        state.er_face[:] = 30.+5*rf+3*z
        state.ez_face[:] = 10.+3*r+2*zf
        calculate_transport_coefficients(state, config)
        radial_norm = np.hypot(state.er_face[1:], 10.+3*rf[1:]+2*z)
        axial_norm = np.hypot(30.+5*r+3*zf[:, 1:], state.ez_face[:, 1:])
        expected_r = 3.74e24*radial_norm**(-.25)/molecular_density(config)
        expected_z = 3.74e24*axial_norm**(-.25)/molecular_density(config)
        assert_allclose(state.mobility_r[1:], expected_r, rtol=1e-14)
        assert_allclose(state.mobility_z[:, 1:], expected_z, rtol=1e-14)
        assert_array_equal(state.vex[0], 0.)

    def test_reactions_use_center_field_and_are_independent_of_face_velocities(self):
        config = self.config()
        state = initialize_simulation(config)
        state.abs_e[:] = 100.
        state.vex[:] = 1e50
        state.vey[:] = -1e50
        calculate_reaction_coefficients(state)
        speed = abs(electron_drift(100., 100.))
        assert_allclose(state.k1, 1.4e-20*np.exp(-6.6)*speed, rtol=1e-14)
        previous = [getattr(state, k).copy() for k in ('k1', 'k2', 'k3')]
        state.vex[:] = state.vey[:] = 0.
        calculate_reaction_coefficients(state)
        for name, values in zip(('k1', 'k2', 'k3'), previous):
            assert_array_equal(getattr(state, name), values)

    def test_discrete_gauss_law_matches_poisson_in_gas_with_electrode(self):
        config = self.config(nr=7, nz=9, electrode_gap=.45, cathode_radius=.2)
        state = initialize_simulation(config)
        rng = np.random.default_rng(51)
        state.phi[:] = rng.normal(size=state.ne.shape)
        voltage = -3.
        state.phi[~state.gas_mask] = voltage
        calculate_electric_field(state, config, voltage=voltage)
        ar, az = state.grid.areas
        flux_r, flux_z = state.er_face*ar, state.ez_face*az
        divergence = (flux_r[1:]-flux_r[:-1]+flux_z[:, 1:]-flux_z[:, :-1])
        divergence *= molecular_density(config)*1e-21/state.volume
        source = discretization(state.rh, state.zh, state.p1, state.p2, state.p3,
            state.p4, state.p5, gas_mask=state.gas_mask,
            radial_boundary=state.radial_boundary, upper_boundary=state.upper_boundary)
        expected = np.divide(state.phi, state.p1, out=np.zeros_like(state.phi), where=state.gas_mask)
        expected[:, :-1] += state.p2[:, :-1]*state.phi[:, 1:]
        expected[:, 1:] += state.p5[:, 1:]*state.phi[:, :-1]
        expected[1:] += state.p3[1:]*state.phi[:-1]
        expected[:-1] += state.p4[:-1]*state.phi[1:]
        expected -= voltage*source
        assert_allclose(divergence[state.gas_mask], expected[state.gas_mask], rtol=2e-13, atol=1e-11)

    def test_electrode_normal_absorption_and_zero_tangential_field(self):
        config = self.config(nr=8, nz=12, delta=1e-4, electrode_gap=6e-4,
                             cathode_radius=3e-4, voltage=500.)
        state = initialize_simulation(config)
        prepare_step(state, config, 0., verbose=False)
        gas = state.gas_mask
        assert_array_equal(state.ex[~gas], 0.)
        assert_array_equal(state.ey[~gas], 0.)
        metal_r, metal_z = ~gas[:-1] & ~gas[1:], ~gas[:, :-1] & ~gas[:, 1:]
        assert_array_equal(state.vex[1:-1][metal_r], 0.)
        assert_array_equal(state.vey[:, 1:-1][metal_z], 0.)
        face_r, face_z = gas[:-1] != gas[1:], gas[:, :-1] != gas[:, 1:]
        scale = molecular_density(config)*1e-21
        assert_allclose(state.mobility_r[1:-1][face_r],
                        3.74e24*np.abs(state.er_face[1:-1][face_r])**(-.25)/molecular_density(config))
        self.assertTrue(np.any(np.abs(state.vey[:, 1:-1][face_z]) > 0.))
        assert_allclose(state.vey[:, 1:-1][face_z],
                        -state.mobility_z[:, 1:-1][face_z]*state.ez_face[:, 1:-1][face_z]*scale)
        state.ne[:] = 1e15*gas
        initial = np.sum(state.ne*state.volume)*2*np.pi
        losses = ElectronTransport(state, config.delta).step(state, .2*transport_limit(state, config))
        self.assertGreater(losses[0], 0.)
        self.assertAlmostEqual((np.sum(state.ne*state.volume)*2*np.pi+losses.sum())/initial, 1., places=13)

    def test_external_face_velocities_are_used_without_copying_inner_faces(self):
        config = self.config(delta=1.)
        state = initialize_simulation(config)
        state.gas_mask[:] = True
        state.ne[:] = 1.
        state.vex[:] = state.vey[:] = 0.
        state.vex[-1] = 2.
        state.vey[:, -1] = 3.
        initial = np.sum(state.ne*state.volume)*2*np.pi
        losses = ElectronTransport(state, config.delta).step(state, .01)
        ar, az = state.grid.areas
        expected = .01*(2*np.sum(ar[-1]) + 3*np.sum(az[:, -1]))*2*np.pi
        self.assertAlmostEqual(losses[2], expected)
        self.assertAlmostEqual(np.sum(state.ne*state.volume)*2*np.pi+losses.sum(), initial)

    def test_incomplete_transport_face_arrays_are_rejected(self):
        state = initialize_simulation(self.config())
        rng = np.random.default_rng(12)
        vr, vz = rng.normal(size=state.ne.shape), rng.normal(size=state.ne.shape)
        with self.assertRaisesRegex(ValueError, 'face shapes'):
            state.grid.pack_faces(vr, vz)
        with self.assertRaises(ValueError):
            state.grid.pack_faces(np.zeros((2, 2)), vz)

    def test_dielectric_limit_uses_actual_face_mobility_and_neighbor_density(self):
        config = self.config()
        state = initialize_simulation(config)
        state.ne[2, 3] = 2e20
        state.mobility_r[:] = state.mobility_z[:] = .1
        state.mobility_r[3, 3] = .7
        state.abs_e[:] = 1e40  # 中心値で移動度を再計算しないことも確認。
        expected = config.vacuum_permittivity/(config.elementary_charge*.7*2e20)
        self.assertAlmostEqual(dielectric_limit(state, config)/expected, 1.)

    def test_smooth_center_field_converges_at_second_order_away_from_outer_boundary(self):
        errors = []
        for size in (8, 16, 32):
            config = self.config(nr=size, nz=2*size, delta=1/size, electrode_gap=3.)
            state = initialize_simulation(config)
            self.prescribe(state, config, lambda r, z: (r*r+.25*r**4)*np.sin(np.pi*z/2)+.3*z)
            r, z = state.rh[:, None], state.zh[None, :]
            er = -(2*r+r**3)*np.sin(np.pi*z/2)
            ez = -(r*r+.25*r**4)*(np.pi/2)*np.cos(np.pi*z/2)-.3
            selected = (slice(None, -2), slice(2, -2))
            error = (state.ex-er)**2+(state.ey-ez)**2
            errors.append(float(np.sqrt(np.mean(error[selected]))))
        self.assertTrue(3. < errors[0]/errors[1] < 5.)
        self.assertTrue(3. < errors[1]/errors[2] < 5.)

    def test_impulse_boundary_field_uses_current_voltage(self):
        config = self.config(nr=8, nz=12, delta=1e-4, electrode_gap=6e-4,
                             cathode_radius=3e-4, voltage=1234.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'triangle.txt'
            path.write_text('0 0\n0.001 1\n0.002 0\n', encoding='utf-8')
            pulse = replace(config, voltage_mode='impulse', voltage_waveform_file=str(path),
                            impulse_peak_voltage=500.)
            left, right = initialize_simulation(pulse), initialize_simulation(replace(config, voltage=500.))
            from streamer_drift_diffusion import load_voltage_waveform
            prepare_step(left, pulse, 1e-12, waveform=load_voltage_waveform(path), verbose=False)
            prepare_step(right, replace(config, voltage=500.), 0., verbose=False)
            for name in ('phi', 'er_face', 'ez_face', 'ex', 'ey', 'abs_e', 'vex', 'vey'):
                assert_allclose(getattr(left, name), getattr(right, name), rtol=2e-13, atol=1e-9)

    def test_csv_fields_remain_centered_and_metadata_describes_layout(self):
        config = self.config(adaptive_timestep=True, t_end=1e-12, output_interval=1e-12)
        with tempfile.TemporaryDirectory() as directory:
            state = run_simulation(config, output_dir=directory, verbose=False)
            metadata = json.loads((Path(directory)/'run_config.json').read_text())
            self.assertEqual(metadata['model'], 'staggered-v1')
            self.assertEqual(metadata['field_unit'], 'Td')
            for prefix, values in (('Ex', state.ex), ('Ey', state.ey), ('absE', state.abs_e)):
                saved = np.loadtxt(Path(directory)/f'{prefix}1.csv').reshape(state.ne.shape)
                assert_allclose(saved, values, rtol=6e-7, atol=1e-12)

    def test_failure_snapshot_contains_complete_face_arrays(self):
        config = self.config(adaptive_timestep=True, t_end=3e-12, adaptive_max_steps=1)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(TimestepError):
                run_simulation(config, output_dir=directory, verbose=False)
            with np.load(Path(directory)/'failure_state.npz') as snapshot:
                self.assertEqual(snapshot['er_face'].shape, (7, 8))
                self.assertEqual(snapshot['vey'].shape, (6, 9))


if __name__ == '__main__':
    unittest.main()
