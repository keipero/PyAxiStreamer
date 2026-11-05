"""FiPy正本・NumPyビュー・求解失敗時の復元と停止診断を検証する。"""
from dataclasses import fields, replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from fipy import CellVariable
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from streamer_drift_diffusion import SimulationConfig, initialize_simulation, ElectronTransport
from streamer_drift_diffusion.electrostatics import calculate_potential
from streamer_drift_diffusion.output import SimulationOutput
from streamer_drift_diffusion.reactions import apply_reactions
from streamer_drift_diffusion.timestepping import advance_density, DENSITY_NAMES, TimestepError


class StateStorageTests(unittest.TestCase):
    def config(self, **changes):
        """共有変数の検査用に、小さい全気体格子の設定を返す。

        Parameters
        ----------
        **changes : object
            検証ごとに変更するSimulationConfigの設定値。

        Returns
        -------
        SimulationConfig
            初期密度・背景供給0の2×3格子。
        """
        defaults = dict(nr=2, nz=3, delta=.1, electrode_gap=1., cathode_radius=.2,
                        voltage=0., nmax=0., background_ionization=0., t_end=0.)
        return SimulationConfig(**(defaults | changes))

    def test_only_fipy_variables_own_current_density_and_potential(self):
        """状態・輸送・Poissonが同じ正本を使い、2D値を二重保持しない。"""
        state = initialize_simulation(self.config())
        transport = ElectronTransport(state, state.grid.spacing)
        self.assertIsInstance(state.ne_var, CellVariable)
        self.assertIs(transport.density, state.ne_var)
        self.assertIs(state.phi_var, state.poisson.potential)
        self.assertNotIn('ne', {f.name for f in fields(state)})
        self.assertNotIn('phi', state.__dict__)
        self.assertTrue(np.shares_memory(state.ne, state.ne_var.value))
        self.assertTrue(np.shares_memory(state.phi, state.phi_var.value))
        values = np.arange(6., dtype=float).reshape((2, 3))
        state.ne_var.setValue(values.ravel(order='F'))
        assert_array_equal(state.ne, values)
        state.phi[1, 2] = 12.
        self.assertEqual(state.phi_var.value[5], 12.)
        other = replace(state, ne_var=CellVariable(mesh=state.grid.mesh, value=0., hasOld=True))
        with self.assertRaisesRegex(ValueError, 'density'):
            transport.prepare(other)

    def test_reactions_update_shared_density_and_notify_cached_fipy_expressions(self):
        """反応が正本を更新し、計算済みのFiPy派生式にも変更を通知する。"""
        config = self.config(background_ionization=4.)
        state = initialize_simulation(config)
        transport = ElectronTransport(state, config.delta)
        expression = 2*state.ne_var
        before = expression.value.copy()
        view = state.ne
        apply_reactions(state, config, dt=config.dt/2)
        assert_array_equal(view, 2.)
        assert_array_equal(transport.density.value, 2.)
        assert_array_equal(expression.value, before+4.)

    def test_transport_exception_restores_the_shared_current_density(self):
        """求解途中で正本を書き換えて例外を送出しても元の密度へ戻る。"""
        state = initialize_simulation(self.config())
        state.ne_var.setValue(np.arange(1., 7.))
        original = state.ne.copy()
        transport = ElectronTransport(state, state.grid.spacing)

        def fail(**kwargs):
            """求解変数を破壊して失敗を模擬する。

            Parameters
            ----------
            **kwargs : object
                FiPy方程式のsolve()引数。検査には使わない。

            Raises
            ------
            RuntimeError
                求解途中の例外を模擬する。
            """
            transport.density.setValue(np.nan)
            raise RuntimeError('solver failure')

        with patch.object(transport.equation, 'solve', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'solver failure'):
                transport.step(state, 1e-12)
        assert_array_equal(state.ne, original)
        assert_array_equal(transport.density.old.value, original.ravel(order='F'))

    def test_terminal_rejection_restores_current_old_and_other_species(self):
        """最終棄却で全密度とFiPy旧値が残り、派生式のキャッシュも復元される。"""
        config = self.config(adaptive_timestep=True, max_step_retries=0)
        state = initialize_simulation(config)
        state.ne_var.setValue(1.)
        before = {name: getattr(state, name).copy() for name in DENSITY_NAMES}
        expression = 2*state.ne_var
        _ = expression.value.copy()

        def fail(current, current_config, *, dt):
            """反応後の状態破壊を模擬する。

            Parameters
            ----------
            current : SimulationState
                破壊を模擬する状態。
            current_config : SimulationConfig
                反応の呼出し設定。検査には使わない。
            dt : float
                実際の試行幅 [s]。検査には使わない。

            Raises
            ------
            TimestepError
                試行棄却を模擬する。
            """
            current.ne_var.setValue(-5.)
            current.o2[:] = -1.
            _ = expression.value.copy()
            raise TimestepError('reaction failure')

        with patch('streamer_drift_diffusion.reactions.apply_reactions', side_effect=fail):
            with self.assertRaisesRegex(TimestepError, 'reaction failure'):
                advance_density(state, config, config.dt)
        for name in DENSITY_NAMES:
            assert_array_equal(getattr(state, name), before[name])
        assert_array_equal(state.ne_var.old.value, before['ne'].ravel(order='F'))
        assert_array_equal(expression.value, 2*before['ne'].ravel(order='F'))
        assert_array_equal(state.electron_losses, 0.)

    def test_poisson_exception_restores_shared_potential_and_allows_recovery(self):
        """Poissonの中途例外で電位を復元し、その後の正常求解も可能とする。"""
        config = self.config(voltage=500.)
        state = initialize_simulation(config)
        calculate_potential(state, config)
        view = state.phi
        original = view.copy()

        def fail(rhs):
            """電位の正本を破壊して求解例外を模擬する。

            Parameters
            ----------
            rhs : ndarray
                SuperLU求解への右辺。検査には使わない。

            Raises
            ------
            RuntimeError
                求解途中の例外を模擬する。
            """
            state.phi_var.setValue(np.nan)
            raise RuntimeError('Poisson failure')

        state.poisson._lu = SimpleNamespace(solve=fail)
        with self.assertRaisesRegex(RuntimeError, 'Poisson failure'):
            calculate_potential(state, config)
        assert_array_equal(view, original)
        calculate_potential(state, config)
        assert_allclose(view, original, rtol=1e-14)

    def test_failure_archive_contains_numeric_two_dimensional_views(self):
        """停止診断にne・phiを保存し、FiPyオブジェクトを混入させない。"""
        config = self.config(voltage=500., nmax=1e12)
        state = initialize_simulation(config)
        calculate_potential(state, config)
        with tempfile.TemporaryDirectory() as temporary:
            with SimulationOutput(temporary, config, state) as output:
                output.save_failure(0, 0., RuntimeError('example'))
            with np.load(Path(temporary)/'failure_state.npz', allow_pickle=False) as archive:
                assert_array_equal(archive['ne'], state.ne)
                assert_array_equal(archive['phi'], state.phi)
                self.assertNotIn('ne_var', archive.files)
                for name in archive.files:
                    self.assertFalse(archive[name].dtype.hasobject)


if __name__ == '__main__':
    unittest.main()
