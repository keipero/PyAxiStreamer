"""二次ストリーマ用の反応機構（追加化学種・解析式の係数）とイオン移動の検証。"""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from streamer_drift_diffusion import SimulationConfig, initialize_simulation, run_simulation
from streamer_drift_diffusion.electrostatics import calculate_charge_density
from streamer_drift_diffusion.kinetics import (MassActionMechanism, analytic_rate,
                                               load_mechanism_definition)
from streamer_drift_diffusion.output import open_fields
from streamer_drift_diffusion.reactions import apply_reactions, calculate_reaction_coefficients
from streamer_drift_diffusion.simulation_types import molecular_density
from streamer_drift_diffusion.timestepping import timestep_limits
from streamer_drift_diffusion.transport import ElectronTransport

MECHANISM = Path(__file__).resolve().parents[1] / "streamer_drift_diffusion" / "data" / "secondary_air_chemistry.yaml"
COMPOSITION = {  # 配列名 -> (N原子, O原子)
    "n2": (2, 0), "o2": (0, 2), "n2p": (2, 0), "o2p": (0, 2), "om": (0, 1), "o2m": (0, 2),
    "o": (0, 1), "n2o": (2, 1), "n_atom": (1, 0), "n4p": (4, 0), "o4p": (0, 4),
    "n2o2p": (2, 2), "o3m": (0, 3), "n2c": (2, 0),
}


def write_table(path, config, fields=(0., 50., 200., 1000., 5000.)):
    """BOLOSを使わずに、機構の解析式と仮の電子衝突係数から形式4の表を書く。"""
    _, reactions, mechanism = load_mechanism_definition(MECHANISM)
    fields = np.asarray(fields)
    mean_energy = .04 + fields / 200.
    columns = []
    for reaction in reactions:
        spec = reaction["rate"]
        if "collision" in spec:
            scale = 1e-38 if spec["collision"].get("density_normalized_cm3") else 1e-16
            columns.append(scale * fields / fields[-1])
        else:
            columns.append(analytic_rate(spec, fields, temperature=config.temperature,
                                         mean_energy_ev=mean_energy))
    density = molecular_density(config)
    data = {
        "schema": 4, "field_td": fields.tolist(), "rates_si": np.column_stack(columns).tolist(),
        "mechanism": mechanism, "mean_energy_ev": mean_energy.tolist(),
        "mobility_n_per_m_v_s": (density * .04 * np.ones_like(fields)).tolist(),
        "diffusion_n_per_m_s": (density * .1 * np.ones_like(fields)).tolist(),
        "metadata": {"temperature": config.temperature, "nitrogen_fraction": config.nitrogen_fraction,
                     "oxygen_fraction": config.oxygen_fraction, "reactions": mechanism["reaction_ids"]},
    }
    Path(path).write_text(json.dumps(data), encoding="utf-8")


class SecondaryMechanismTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.config = SimulationConfig(
            nr=4, nz=6, delta=1e-4, electrode_gap=3e-4, cathode_radius=1e-4, voltage=500.,
            nmax=1e15, z0=2e-4, background_ionization=0., electron_diffusion=0.,
            chemistry_table=self.folder / "rates.json", t_end=2e-12, output_interval=1e-12)
        write_table(self.config.chemistry_table, self.config)

    def test_mechanism_declares_species_and_charges(self):
        _, reactions, data = load_mechanism_definition(MECHANISM)
        mechanism = MassActionMechanism(data, tabulated=True)
        self.assertEqual(mechanism.density_names[9:], ("n_atom", "n4p", "o4p", "n2o2p", "o3m", "n2c"))
        self.assertEqual(mechanism.charges["n4p"], 1)
        self.assertEqual(mechanism.charges["o3m"], -1)
        self.assertNotIn("n2c", mechanism.charges)
        self.assertEqual(len(set(data["reaction_ids"])), len(reactions))

    def test_analytic_rates_reproduce_the_tabulated_formulas(self):
        # X1: 5.0e-29 (300/T)^2 cm6/s at 300 K
        assert_allclose(analytic_rate({"arrhenius": {"A": 4.5e-36, "b": -2.}}, 0., temperature=300.,
                                      mean_energy_ev=1.), 5e-41, rtol=1e-12)
        # X12: detachment form equals the earlier exp_v1 fit
        field = np.array([0., 100., 1000.])
        expected = 1.24e-17 * np.exp(-(179. / (8.8 + field))**2)
        assert_allclose(analytic_rate({"field": {"A": 1.24e-17, "B": 179., "C": 8.8}}, field,
                                      temperature=300., mean_energy_ev=1.), expected, rtol=1e-14)
        # R10: 2.8e-13 sqrt(300/Te); Te = 2/3 <eps>/kB
        mean_energy = 1.5 * 1.380649e-23 * 3000. / 1.602176634e-19
        assert_allclose(analytic_rate({"electron_temperature": {"A": 4.8497422611928575e-12, "b": -.5}},
                                      0., temperature=300., mean_energy_ev=mean_energy),
                        2.8e-13 * np.sqrt(300. / 3000.), rtol=1e-12)
        with self.assertRaises(ValueError):
            analytic_rate({"constant": -1.}, 0., temperature=300., mean_energy_ev=1.)

    def test_extra_species_react_with_charge_and_atom_conservation(self):
        state = initialize_simulation(self.config)
        gas = state.gas_mask
        self.assertEqual(set(state.extra_species), {"n_atom", "n4p", "o4p", "n2o2p", "o3m", "n2c"})
        rng = np.random.default_rng(5)
        for name in state.density_names:
            if name not in ("n2", "o2"):
                getattr(state, name)[:] = rng.uniform(1e17, 1e19, gas.shape) * gas
        state.abs_e[gas] = 150.
        calculate_reaction_coefficients(state)

        def totals():
            charge = sum(c * getattr(state, name) for name, c in state.charges.items())
            nitrogen = sum(n * getattr(state, name) for name, (n, _) in COMPOSITION.items())
            oxygen = sum(o * getattr(state, name) for name, (_, o) in COMPOSITION.items())
            return charge, nitrogen, oxygen

        before = totals()
        apply_reactions(state, self.config, dt=1e-12)
        for initial, final in zip(before, totals()):
            assert_allclose(final[gas], initial[gas], rtol=1e-12, atol=1e-12 * np.abs(initial).max())
        self.assertTrue(np.all(state.n4p[gas] > 0.))
        # 電荷密度はクラスタイオンも含む
        for name in state.density_names:
            getattr(state, name)[:] = 0.
        state.n4p[gas], state.o3m[gas] = 3e18, 1e18
        calculate_charge_density(state, self.config)
        scale = self.config.elementary_charge / self.config.vacuum_permittivity
        assert_allclose(state.rho[gas], scale * 2e18, rtol=1e-14)

    def test_run_saves_extra_species_and_metadata(self):
        output = self.folder / "output"
        state = run_simulation(self.config, output_dir=output, verbose=False)
        frame = open_fields(output).isel(time=-1).load()
        for name, species in (("n4p", "N4+"), ("o3m", "O3-"), ("n2c", "N2(C)")):
            self.assertEqual(frame[name].attrs["species"], species)
            assert_array_equal(frame[name], getattr(state, name))
        fields = open_fields(output)
        self.assertGreater(fields.sizes["time"], 1)
        self.assertEqual(json.loads(fields.attrs["run_config"])["config"]["nr"], self.config.nr)


class IonDriftTests(unittest.TestCase):
    def setUp(self):
        self.config = SimulationConfig(nr=4, nz=8, delta=1e-4, electrode_gap=1., voltage=0., nmax=0.,
                                       background_ionization=0., positive_ion_mobility=2e-4,
                                       negative_ion_mobility=3e-4)
        self.state = initialize_simulation(self.config)
        self.density = molecular_density(self.config)
        self.td = 100.  # 軸方向の一様電場 [Td]

    def transport(self):
        self.state.ez_face[:] = self.td
        return ElectronTransport(self.state, self.config.delta, gas_density=self.density).prepare(self.state)

    def test_mobilities_are_assigned_by_charge_sign(self):
        mobilities = {name: (charge, mobility) for name, charge, mobility in self.state.ion_mobilities}
        self.assertEqual(mobilities["n2p"], (1, 2e-4))
        self.assertEqual(mobilities["o2m"], (-1, 3e-4))
        self.assertNotIn("ne", mobilities)
        no_ions = initialize_simulation(replace(self.config, positive_ion_mobility=0., negative_ion_mobility=0.))
        self.assertEqual(no_ions.ion_mobilities, ())
        with self.assertRaises(ValueError):
            ElectronTransport(self.state, self.config.delta)
        with self.assertRaises(ValueError):
            initialize_simulation(replace(self.config, positive_ion_mobility=-1.))

    def test_upwind_drift_moves_ions_along_and_against_the_field(self):
        transport = self.transport()
        speed_p = 2e-4 * self.td * 1e-21 * self.density
        speed_n = 3e-4 * self.td * 1e-21 * self.density
        assert_allclose(transport.ion_limit, self.config.delta / speed_n, rtol=1e-12)
        self.assertLessEqual(transport.transport_limit, transport.ion_limit)
        self.state.n2p[:, 3] = 1e18
        self.state.o2m[:, 4] = 1e18
        dt = .5 * transport.ion_limit
        before_p, before_n = self.state.n2p.sum(), self.state.o2m.sum()
        transport.step(self.state, dt, prepared=True)
        fraction_p, fraction_n = speed_p * dt / self.config.delta, speed_n * dt / self.config.delta
        assert_allclose(self.state.n2p[:, 3], 1e18 * (1 - fraction_p), rtol=1e-12)
        assert_allclose(self.state.n2p[:, 4], 1e18 * fraction_p, rtol=1e-12)
        assert_allclose(self.state.o2m[:, 4], 1e18 * (1 - fraction_n), rtol=1e-12)
        assert_allclose(self.state.o2m[:, 3], 1e18 * fraction_n, rtol=1e-12)
        assert_allclose(self.state.n2p.sum(), before_p, rtol=1e-14)
        assert_allclose(self.state.o2m.sum(), before_n, rtol=1e-14)

    def test_ions_leave_through_absorbing_boundaries_without_inflow(self):
        transport = self.transport()
        self.state.n2p[:, -1] = 1e18   # 上端へ流出
        self.state.o2m[:, -1] = 1e18   # 上端から下向きに移動（流入はない）
        dt = .5 * transport.ion_limit
        transport.step(self.state, dt, prepared=True)
        fraction_p = 2e-4 * self.td * 1e-21 * self.density * dt / self.config.delta
        assert_allclose(self.state.n2p[:, -1], 1e18 * (1 - fraction_p), rtol=1e-12)
        self.assertLess(self.state.n2p.sum(), 1e18 * self.state.n2p.shape[0])
        assert_allclose(self.state.o2m.sum(), 1e18 * self.state.o2m.shape[0], rtol=1e-14)

    def test_time_step_limit_includes_ion_drift(self):
        self.state.ez_face[:] = self.td
        limits = timestep_limits(self.state, self.config)
        transport = self.transport()
        self.assertLessEqual(limits.cfl, self.config.dt_cfl_number * transport.ion_limit * (1 + 1e-12))


if __name__ == "__main__":
    unittest.main()
