"""係数表から空間連成までの反応収支と入力検査。"""
from dataclasses import replace
from importlib.util import find_spec
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from streamer_drift_diffusion import SimulationConfig, initialize_simulation, run_simulation
from streamer_drift_diffusion.kinetics import DENSITY_NAMES, MECHANISM_PATH, export_mechanism
from streamer_drift_diffusion.plasma_chemistry import LEGACY_EQUATIONS, load_reaction_table
from streamer_drift_diffusion.reactions import apply_reactions, calculate_reaction_coefficients
from streamer_drift_diffusion.simulation_types import molecular_density
from streamer_drift_diffusion.timestepping import chemistry_limit, timestep_limits
from streamer_drift_diffusion.transport_coefficients import calculate_transport_coefficients


class ChemistryTests(unittest.TestCase):
    """検証用係数表で異なる電離係数と中性Oの収支を確認する。"""

    def setUp(self):
        """実測の断面積ファイルから独立した小さな表を用意する。"""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        folder = Path(temporary.name)
        self.path = folder / "rates.json"
        self.output = folder / "output"
        self.config = SimulationConfig(
            nr=4, nz=6, delta=1e-4, electrode_gap=3e-4, cathode_radius=1e-4,
            voltage=500., nmax=1e15, z0=2e-4, background_ionization=0.,
            chemistry_table=self.path, t_end=3e-12,
        )
        data = {
            "schema": 1,
            "field_td": [0., 100., 1000.],
            "rates_m3_s": [[0., 0., 0.], [1e-17, 3e-17, 2e-17], [1e-16, 3e-16, 2e-16]],
            "metadata": {
                "temperature": 300., "nitrogen_fraction": .8, "oxygen_fraction": .2,
                "reactions": list(LEGACY_EQUATIONS),
            },
        }
        self.path.write_text(json.dumps(data), encoding="utf-8")

    def add_swarm_coefficients(self):
        """分子数密度で割ると既知の値になる輸送係数列を追加する。"""
        data = json.loads(self.path.read_text(encoding="utf-8"))
        density = molecular_density(self.config)
        data.update(schema=2, mobility_n_per_m_v_s=(density*np.array([2., 1., .5])).tolist(),
                    diffusion_n_per_m_s=(density*np.array([.02, .1, 1.])).tolist())
        self.path.write_text(json.dumps(data), encoding="utf-8")

    def test_rejects_mismatched_gas_conditions_and_field_outside_table(self):
        """表の組成・温度を照合し、範囲外の電場で停止する。"""
        with self.assertRaisesRegex(ValueError, "temperature"):
            load_reaction_table(self.path, replace(self.config, temperature=301.))
        table = load_reaction_table(self.path, self.config)
        assert_allclose(table.sample([50.]).ravel(), [5e-18, 1.5e-17, 1e-17], rtol=1e-14)
        with self.assertRaisesRegex(RuntimeError, "exceeds table maximum"):
            table.sample([1001.])

    def test_reaction_step_conserves_charge_and_elements_with_separate_ionization(self):
        """全4反応の数密度更新を局所的に検査する。"""
        state = initialize_simulation(self.config)
        gas = state.gas_mask
        state.abs_e[gas] = 100.
        state.ne_var.setValue(np.where(gas, 1e18, 0.).ravel(order="F"))
        state.n2[gas], state.o2[gas] = 1e20, 2e20
        state.n2p[gas], state.o2p[gas] = 2e18, 0.
        calculate_reaction_coefficients(state)
        state.k3[gas] = 1e-38
        dt = 1e-6
        limit = chemistry_limit(state, self.config)
        self.assertLessEqual(limit, self.config.dt_chemistry_fraction / (3e-17*2e20))
        apply_reactions(state, self.config, dt=dt)
        assert_allclose(state.n2p[gas] - 2e18, 1e15, rtol=1e-13)
        assert_allclose(state.o2p[gas], 6e15, rtol=1e-13)
        assert_allclose(state.o[gas], 4e15, rtol=1e-13)
        assert_array_equal(state.om, state.o)
        charge = state.n2p + state.o2p - state.om - state.o2m - state.ne
        nitrogen = 2*(state.n2 + state.n2p)
        oxygen = 2*(state.o2 + state.o2p + state.o2m) + state.om + state.o
        assert_allclose(charge[gas], 1e18, rtol=1e-15)
        assert_allclose(nitrogen[gas], 2.04e20, rtol=1e-15)
        assert_allclose(oxygen[gas], 4e20, rtol=1e-15)
        assert_array_equal(state.o[~gas], 0.)

    def test_grid_run_records_table_and_saves_atomic_oxygen(self):
        """通常の時間ループで係数表を使用し、出力へ由来を記録する。"""
        self.add_swarm_coefficients()
        state = run_simulation(self.config, output_dir=self.output, verbose=False)
        self.assertGreater(float(state.o.max()), 0.)
        assert_array_equal(state.o, state.om)
        assert_allclose(np.loadtxt(self.output/"O1.csv").reshape(state.o.shape), state.o, rtol=5e-7)
        report = json.loads((self.output/"run_config.json").read_text(encoding="utf-8"))
        self.assertEqual(report["chemistry_backend"], "py3bolos-cantera-table")
        self.assertEqual(report["electron_mobility_source"], "py3bolos-table")
        self.assertEqual(report["electron_diffusion_source"], "py3bolos-table")
        self.assertEqual(report["chemistry_table_metadata"]["max_field_td"], 1000.)
        self.assertEqual(len(report["chemistry_table_metadata"]["table_sha256"]), 64)

    @unittest.skipUnless(find_spec("cantera") is not None, "Cantera is required")
    def test_mechanism_matches_cantera_zero_dimensional_source(self):
        """全4反応の格子更新をCanteraの単一セルの生成速度と照合する。"""
        import cantera as ct

        phase = ct.Solution(str(MECHANISM_PATH))
        reactions, mechanism = export_mechanism(phase)
        self.add_swarm_coefficients()
        data = json.loads(self.path.read_text(encoding="utf-8"))
        data.update(schema=3, mechanism=mechanism)
        self.path.write_text(json.dumps(data), encoding="utf-8")
        state = initialize_simulation(self.config)
        phase.TPX = 300., 10., {"N2": .3, "O2": .6, "Electron": .01, "N2+": .02}
        initial = phase.concentrations * ct.avogadro
        mask = state.gas_mask
        state.ne_var.setValue(np.where(mask, initial[phase.species_index("Electron")], 0.).ravel(order="F"))
        for name, species in zip(DENSITY_NAMES[1:], mechanism["species"][1:]):
            getattr(state, name)[mask] = initial[phase.species_index(species)]
        state.abs_e[mask] = 100.
        calculate_reaction_coefficients(state)
        state.k3[mask] = 1e-38
        indices = {reaction.ID: index for index, reaction in enumerate(phase.reactions())}
        for reaction, orders, k in zip(reactions, mechanism["orders"],
                                       (state.k1, state.k1_o2, state.k2, state.k3)):
            phase.set_multiplier(k[mask][0] * ct.avogadro**(sum(orders) - 1),
                                 indices[reaction.ID])
        limit = chemistry_limit(state, self.config)
        legacy = replace(state, chemistry=replace(state.chemistry, mechanism=None))
        self.assertAlmostEqual(limit / chemistry_limit(legacy, self.config), 1.)
        dt = limit / 2
        expected = initial + dt * phase.net_production_rates * ct.avogadro
        apply_reactions(state, self.config, dt=dt)
        for name, species in zip(DENSITY_NAMES, mechanism["species"]):
            density = getattr(state, name)
            assert_allclose(density[mask], expected[phase.species_index(species)], rtol=5e-15)
            assert_array_equal(density[~mask], 0.)

    def test_swarm_coefficients_at_zero_field_and_signed_face_field(self):
        """ゼロ電場の拡散、面速度の符号、中心Dの補間と刻み制約を確認する。"""
        self.add_swarm_coefficients()
        state = initialize_simulation(self.config)
        assert_array_equal(state.vex, 0.)
        assert_array_equal(state.vey, 0.)
        assert_allclose(state.de[state.gas_mask], .02, rtol=1e-14)
        self.assertTrue(np.isfinite(timestep_limits(state, self.config).diffusion))
        state.ez_face[0, 1] = -100.
        state.abs_e[state.gas_mask] = 50.
        calculate_transport_coefficients(state, self.config)
        assert_allclose(state.mobility_z[0, 1], 1., rtol=1e-14)
        assert_allclose(state.vey[0, 1], molecular_density(self.config)*100e-21, rtol=1e-14)
        assert_allclose(state.de[state.gas_mask], .06, rtol=1e-14)
        assert_array_equal(state.de[~state.gas_mask], 0.)
        assert_array_equal(state.mobility_r[0], 0.)

    def test_swarm_table_rejects_a_constant_diffusion_override(self):
        """BOLOS拡散を指定した計算で、一定拡散との併記を検出する。"""
        self.add_swarm_coefficients()
        with self.assertRaisesRegex(ValueError, "electron_diffusion=0"):
            initialize_simulation(replace(self.config, electron_diffusion=.1))

    def test_zero_voltage_swarm_diffusion_does_not_stall_at_weak_field(self):
        """拡散で発生する微弱電場で三体付着が刻みを極端に縮める回帰を検出する。"""
        self.add_swarm_coefficients()
        config = replace(self.config, voltage=0., adaptive_timestep=True, max_steps=10)
        state = run_simulation(config, write_output=False, verbose=False)
        self.assertGreater(float(state.electron_losses.sum()), 0.)
        self.assertGreater(float(state.abs_e.max()), 0.)
