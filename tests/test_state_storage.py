"""共有の状態配列・求解失敗時の復元と停止診断を検証する。"""
from dataclasses import fields, replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from streamer_drift_diffusion.output import open_failure, open_fields, open_history
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

    def test_state_transport_and_poisson_share_single_arrays(self):
        """状態・輸送・Poissonが同じ配列を使い、2D値を二重保持しない。"""
        state = initialize_simulation(self.config())
        transport = ElectronTransport(state, state.grid.spacing)
        self.assertIs(transport.density, state.ne)
        self.assertIs(state.phi, state.poisson.potential)
        self.assertIn('ne', {f.name for f in fields(state)})
        state.phi[1, 2] = 12.
        self.assertEqual(state.poisson.potential[1, 2], 12.)
        other = replace(state, ne=np.zeros_like(state.ne))
        with self.assertRaisesRegex(ValueError, 'density'):
            transport.prepare(other)

    def test_reactions_update_shared_density_in_place(self):
        """反応が電子密度をその場で更新し、輸送が同じ値を参照する。"""
        config = self.config(background_ionization=4.)
        state = initialize_simulation(config)
        transport = ElectronTransport(state, config.delta)
        view = state.ne
        apply_reactions(state, config, dt=config.dt/2)
        assert_array_equal(view, 2.)
        assert_array_equal(transport.density, 2.)

    def test_transport_exception_keeps_the_current_density(self):
        """輸送カーネルが例外を送出しても電子密度を書き換えない。"""
        state = initialize_simulation(self.config())
        state.ne[:] = np.arange(1., 7.).reshape(state.ne.shape, order='F')
        original = state.ne.copy()
        transport = ElectronTransport(state, state.grid.spacing)
        with patch('streamer_drift_diffusion.transport.transport_update',
                   side_effect=RuntimeError('solver failure')):
            with self.assertRaisesRegex(RuntimeError, 'solver failure'):
                transport.step(state, 1e-12)
        assert_array_equal(state.ne, original)

    def test_terminal_rejection_restores_all_species(self):
        """最終棄却で電子を含む全密度が開始時の値に戻る。"""
        config = self.config(adaptive_timestep=True, max_step_retries=0)
        state = initialize_simulation(config)
        state.ne[:] = 1.
        before = {name: getattr(state, name).copy() for name in DENSITY_NAMES}

        def fail(current, current_config, *, dt):
            """反応後の状態破壊を模擬し、試行棄却の例外を送出する。"""
            current.ne[:] = -5.
            current.o2[:] = -1.
            current.o[:] = 7.
            raise TimestepError('reaction failure')

        with patch('streamer_drift_diffusion.reactions.apply_reactions', side_effect=fail):
            with self.assertRaisesRegex(TimestepError, 'reaction failure'):
                advance_density(state, config, config.dt)
        for name in DENSITY_NAMES:
            assert_array_equal(getattr(state, name), before[name])
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
            state.phi[:] = np.nan
            raise RuntimeError('Poisson failure')

        state.poisson._lu = SimpleNamespace(solve=fail)
        with self.assertRaisesRegex(RuntimeError, 'Poisson failure'):
            calculate_potential(state, config)
        assert_array_equal(view, original)
        calculate_potential(state, config)
        assert_allclose(view, original, rtol=1e-14)

    def test_failure_archive_contains_numeric_two_dimensional_views(self):
        """停止診断にne・phiを保存し、オブジェクト配列を混入させない。"""
        config = self.config(voltage=500., nmax=1e12)
        state = initialize_simulation(config)
        calculate_potential(state, config)
        with tempfile.TemporaryDirectory() as temporary:
            with SimulationOutput(temporary, config, state) as output:
                output.save_failure(0, 0., RuntimeError('example'))
            archive = open_failure(temporary).load()
            self.assertEqual(archive.attrs['error'], 'example')
            assert_array_equal(archive['ne'], state.ne)
            assert_array_equal(archive['phi'], state.phi)
            for name in archive.data_vars:
                self.assertFalse(archive[name].dtype.hasobject)


if __name__ == '__main__':
    unittest.main()
