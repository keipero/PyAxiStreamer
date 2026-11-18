"""xarray・Zarrによる計算出力（場・刻み履歴・停止診断）の保存と読込み。

出力フォルダには次の3つのZarrストアと設定のJSONを置く。

- ``fields.zarr``: 次元(time, r, z)の場と全化学種の密度。出力時刻ごとにtime方向へ追記する。
- ``history.zarr``: 次元(step,)の受理更新ごとの刻み・制約・収支。
- ``failure.zarr``: 停止時だけ作る、全数値配列の診断スナップショット。
- ``run_config.json``: 計算条件と数値手法の説明。``fields.zarr``の属性にも同じ内容を持つ。

読込みはopen_fields()・open_history()・open_failure()、またはxarray.open_zarr()で行う。
"""
from dataclasses import asdict, fields as dataclass_fields
import json
from pathlib import Path

import numpy as np
import scipy
import xarray as xr

from .kinetics import DRY_AIR_SPECIES_FIELDS


FIELDS_STORE, HISTORY_STORE, FAILURE_STORE = "fields.zarr", "history.zarr", "failure.zarr"

# 保存する場: 変数名 -> (状態の属性名, 単位, 説明)。
FIELD_VARIABLES = {
    "phi": ("phi", "V", "total electric potential"),
    "ex": ("ex", "Td", "cell-centered reduced radial field E_r/N"),
    "ey": ("ey", "Td", "cell-centered reduced axial field E_z/N"),
    "abs_e": ("abs_e", "Td", "cell-centered reduced field magnitude |E|/N"),
    "photoionization": (None, "m-3 s-1", "photoionization source rate"),
}

HISTORY_VARIABLES = ("time", "dt", "dt_cfl", "dt_dielectric", "dt_chemistry", "dt_diffusion",
                     "limiting", "retries", "min_ne", "max_ne", "max_field_td",
                     "electron_number", "charge", "electrode_electron_loss",
                     "ground_electron_loss", "outer_electron_loss")

_ZARR = dict(consolidated=False)


def run_metadata(config, state):
    """計算条件と数値手法の説明を、JSONへ書ける辞書として返す。

    Parameters
    ----------
    config : SimulationConfig
        計算条件。パスは文字列にする。
    state : SimulationState
        係数表・化学機構の有無を参照する状態。

    Returns
    -------
    dict
        ``config``に設定、その他のキーに手法・ライブラリ版・境界条件の説明を持つ。

    Examples
    --------
    >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
    >>> config = SimulationConfig(nr=2, nz=3, t_end=0.)
    >>> run_metadata(config, initialize_simulation(config))['poisson_backend']
    'scipy.sparse'
    """
    settings = {key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(config).items()}
    chemistry = state.chemistry
    use_swarm = chemistry is not None and chemistry.swarm is not None
    neumann = config.flux_neumann_boundaries
    return {"config": settings, "numpy_version": np.__version__, "scipy_version": scipy.__version__,
            "xarray_version": xr.__version__,
            "poisson_backend": "scipy.sparse", "poisson_solver": "SuperLU",
            "poisson_solver_suite": "scipy", "poisson_matrix_reused": True,
            "poisson_factorization_reused": True,
            "poisson_criterion": "max_abs(A*phi-b) / max_abs(b)",
            "transport_backend": "numba", "transport_time_scheme": "explicit Euler",
            "transport_advection_scheme": "van Leer",
            "transport_boundary": ("zero normal gradient of electron flux; zero radial flux on axis"
                                   if neumann else "half-cell absorption sink; no inflow"),
            "model": "staggered-v1", "electrode_boundary": "staircase Dirichlet",
            "field_layout": "cell-centered ex/ey/abs_e; complete face fields and velocities in state",
            "field_unit": "Td",
            "electron_mobility_source": "py3bolos-table" if use_swarm else "empirical",
            "electron_diffusion_source": "py3bolos-table" if use_swarm else "constant verification coefficient",
            "chemistry_backend": ("empirical" if chemistry is None else
                                  "py3bolos-structured-table" if state.reaction_rates is not None else
                                  "py3bolos-cantera-table"),
            "photoionization_model": "bourdon-2007-helmholtz-3",
            "chemistry_table_metadata": None if chemistry is None else {
                **chemistry.metadata, "max_field_td": float(chemistry.field_td[-1])},
            "outer_potential": ("zero normal gradient at upper and radial outer faces"
                                if neumann else "analytic Dirichlet voltage at outer faces"),
            "output_format": "xarray/zarr", "field_and_density_times": "time"}


def frame_dataset(state, step, time, dt):
    """現在の状態から、time長1の場のDatasetを作る。

    Parameters
    ----------
    state : SimulationState
        保存する状態。値はコピーする。
    step : int
        受理済み更新数。
    time : float
        密度・場の時刻 [s]。
    dt : float
        直前の受理刻み [s]。初期フレームは0。

    Returns
    -------
    xarray.Dataset
        次元(time, r, z)。座標はtime [s]・r・z [m]（セル中心）とstep・dt、
        静的な変数にgas_maskとvolume（2πを省略）を持つ。
        密度は全化学種を状態の属性名で保存し、属性speciesに化学種名を持つ。

    Examples
    --------
    >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
    >>> state = initialize_simulation(SimulationConfig(nr=2, nz=3, t_end=0.))
    >>> ds = frame_dataset(state, 0, 0., 0.)
    >>> ds.ne.dims, ds.ne.attrs['units']
    (('time', 'r', 'z'), 'm-3')
    """
    dims = ("time", "r", "z")
    variables = {}
    for species, name in DRY_AIR_SPECIES_FIELDS.items():
        variables[name] = (dims, getattr(state, name)[None].copy(),
                           {"units": "m-3", "species": species, "long_name": f"{species} number density"})
    for name, (attribute, units, description) in FIELD_VARIABLES.items():
        values = state.photoionization.rate if attribute is None else getattr(state, attribute)
        variables[name] = (dims, np.array(values, dtype=float)[None], {"units": units, "long_name": description})
    variables["gas_mask"] = (("r", "z"), state.gas_mask.copy(), {"long_name": "gas cell (False: electrode)"})
    variables["volume"] = (("r", "z"), state.volume.copy(), {"units": "m3", "long_name": "cell volume / 2pi"})
    return xr.Dataset(variables, coords={
        "time": ("time", [float(time)], {"units": "s"}),
        "step": ("time", [int(step)], {"long_name": "accepted steps"}),
        "dt": ("time", [float(dt)], {"units": "s", "long_name": "last accepted timestep"}),
        "r": ("r", state.rh.copy(), {"units": "m", "long_name": "radial cell center"}),
        "z": ("z", state.zh.copy(), {"units": "m", "long_name": "axial cell center"}),
    })


def state_dataset(state):
    """状態の全数値配列を、形状から次元名を付けたDatasetにする。

    Parameters
    ----------
    state : SimulationState
        保存する状態。LU分解・格子などの数値配列以外の属性は含めない。

    Returns
    -------
    xarray.Dataset
        中心(r,z)、半径面(r_face,z)、軸方向面(r,z_face)の配列と、その他の配列。
        その他の配列には``<名前>_dim<軸>``の次元を付ける。

    Examples
    --------
    >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
    >>> state = initialize_simulation(SimulationConfig(nr=2, nz=3, t_end=0.))
    >>> state_dataset(state).vex.dims
    ('r_face', 'z')
    """
    nr, nz = state.grid.shape
    named = {(nr, nz): ("r", "z"), (nr+1, nz): ("r_face", "z"), (nr, nz+1): ("r", "z_face")}
    variables = {}
    for field in dataclass_fields(state):
        values = getattr(state, field.name)
        if isinstance(values, np.ndarray):
            dims = named.get(values.shape) if values.ndim == 2 else None
            dims = dims or tuple(f"{field.name}_dim{axis}" for axis in range(values.ndim))
            variables[field.name] = (dims, values.copy())
    return xr.Dataset(variables)


class SimulationOutput:
    """1回の計算の場・刻み履歴・停止診断を、出力フォルダのZarrストアへ保存する。

    Parameters
    ----------
    directory : str or os.PathLike
        出力フォルダ。なければ作る。既存の``fields.zarr``があればエラー。
    config : SimulationConfig
        計算条件。run_config.jsonと場の属性へ保存する。
    state : SimulationState
        保存する状態。各保存時点の値をコピーする。

    Notes
    -----
    コンテキストマネージャとして使う。場はsave_frame()ごとにtime方向へ追記するため、
    計算が途中で止まってもそれまでのフレームは読める。
    record_step()の履歴はメモリに溜め、save_frame()・save_failure()・終了時にまとめて追記する。

    Examples
    --------
    >>> import tempfile
    >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
    >>> config = SimulationConfig(nr=2, nz=3, t_end=0.)
    >>> state = initialize_simulation(config)
    >>> with tempfile.TemporaryDirectory() as directory:
    ...     with SimulationOutput(directory, config, state) as output:
    ...         output.save_frame(0, 0., 0.)
    ...     open_fields(directory).sizes['time']
    1
    """

    def __init__(self, directory, config, state):
        self.directory = Path(directory)
        self.config = config
        self.state = state
        self.frame = 0
        self._history = {name: [] for name in ("step", *HISTORY_VARIABLES)}
        self._history_written = False

    def __enter__(self):
        if (self.directory / FIELDS_STORE).exists():
            raise FileExistsError(f"output folder already contains {FIELDS_STORE}: {self.directory}")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.metadata = run_metadata(self.config, self.state)
        (self.directory / "run_config.json").write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")
        return self

    def record_step(self, step, time, dt, limits, limiting, retries):
        """受理した1更新の刻み・制約・収支を履歴へ加える。

        Parameters
        ----------
        step : int
            受理済み更新数。
        time, dt : float
            更新後の時刻と受理刻み [s]。
        limits : TimestepLimits
            更新前に評価した各物理上限 [s]。
        limiting : str
            刻みを決めた条件名。再試行時はretry。
        retries : int
            棄却して再試行した回数。

        Examples
        --------
        >>> import tempfile
        >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
        >>> from streamer_drift_diffusion.timestepping import timestep_limits
        >>> config = SimulationConfig(nr=2, nz=3, t_end=0.)
        >>> state = initialize_simulation(config)
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     with SimulationOutput(directory, config, state) as output:
        ...         output.record_step(1, 1e-12, 1e-12, timestep_limits(state, config), 'dt_max', 0)
        ...     open_history(directory).limiting.values.tolist()
        ['dt_max']
        """
        state = self.state
        volume = state.volume * (2 * np.pi)
        charge_density = self.config.elementary_charge * (
            state.n2p + state.o2p - state.om - state.o2m - state.ne)
        row = (step, time, dt, limits.cfl, limits.dielectric, limits.chemistry, limits.diffusion,
               limiting, retries, float(np.min(state.ne[state.gas_mask])), float(np.max(state.ne)),
               float(np.max(state.abs_e)), float(np.sum(state.ne * volume)),
               float(np.sum(charge_density * volume)), *state.electron_losses)
        for values, value in zip(self._history.values(), row):
            values.append(value)

    def _flush_history(self):
        """溜めた履歴をhistory.zarrへ追記し、バッファを空にする。"""
        if not self._history["step"]:
            return
        history = self._history
        units = {"time": "s", "charge": "C", "max_field_td": "Td", "min_ne": "m-3", "max_ne": "m-3"}
        dataset = xr.Dataset(
            {name: ("step", np.array(history[name], dtype=object if name == "limiting" else None),
                    {"units": units.get(name, "s" if name.startswith("dt") else "")})
             for name in HISTORY_VARIABLES},
            coords={"step": ("step", np.array(history["step"], dtype=np.int64))})
        dataset["limiting"].attrs = {"long_name": "condition that set dt"}
        store = self.directory / HISTORY_STORE
        if self._history_written:
            dataset.to_zarr(store, append_dim="step", **_ZARR)
        else:
            dataset.to_zarr(store, mode="w", **_ZARR)
            self._history_written = True
        for values in history.values():
            values.clear()

    def save_frame(self, step, time, dt):
        """現在の状態を1フレームとしてfields.zarrへ追記する。

        Parameters
        ----------
        step : int
            受理済み更新数。
        time : float
            密度・場の時刻 [s]。
        dt : float
            直前の受理刻み [s]。初期フレームは0。
        """
        dataset = frame_dataset(self.state, step, time, dt)
        store = self.directory / FIELDS_STORE
        if self.frame:
            dataset.drop_vars(["gas_mask", "volume", "r", "z"]).to_zarr(store, append_dim="time", **_ZARR)
        else:
            dataset.attrs["run_config"] = json.dumps(self.metadata)
            dataset.to_zarr(store, mode="w", encoding={"time": {"dtype": "float64"}}, **_ZARR)
        self.frame += 1
        self._flush_history()

    def save_failure(self, step, time, exc):
        """停止時の全数値配列と原因をfailure.zarrへ保存する。

        Parameters
        ----------
        step : int
            停止までに受理した更新数。
        time : float
            停止時刻 [s]。
        exc : BaseException
            停止の原因。文字列を属性errorへ保存する。
        """
        self._flush_history()
        dataset = state_dataset(self.state)
        dataset.attrs.update(step=int(step), time=float(time), error=str(exc))
        dataset.to_zarr(self.directory / FAILURE_STORE, mode="w", **_ZARR)

    def __exit__(self, *exc):
        self._flush_history()


def open_fields(directory):
    """出力フォルダの場のDatasetを遅延読込みで開く。

    Parameters
    ----------
    directory : str or os.PathLike
        run_simulation()の出力フォルダ。

    Returns
    -------
    xarray.Dataset
        次元(time, r, z)。属性run_configに計算条件のJSON文字列を持つ。

    Examples
    --------
    >>> import tempfile
    >>> from streamer_drift_diffusion import SimulationConfig, run_simulation
    >>> config = SimulationConfig(nr=4, nz=6, delta=1e-4, electrode_gap=3e-4, t_end=2e-12,
    ...                           output_interval=1e-12)
    >>> with tempfile.TemporaryDirectory() as directory:
    ...     _ = run_simulation(config, output_dir=directory, verbose=False)
    ...     ds = open_fields(directory)
    ...     ds.time.values.tolist(), ds.step.values.tolist()
    ([0.0, 1e-12, 2e-12], [0, 1, 2])
    """
    return xr.open_zarr(Path(directory) / FIELDS_STORE, **_ZARR)


def open_history(directory):
    """受理更新ごとの履歴を開く。更新がなければstep長0のDatasetを返す。

    Parameters
    ----------
    directory : str or os.PathLike
        run_simulation()の出力フォルダ。

    Returns
    -------
    xarray.Dataset
        次元(step,)。変数はHISTORY_VARIABLES。
    """
    store = Path(directory) / HISTORY_STORE
    if store.exists():
        return xr.open_zarr(store, **_ZARR)
    return xr.Dataset({name: ("step", np.array([], dtype=object if name == "limiting" else float))
                       for name in HISTORY_VARIABLES},
                      coords={"step": ("step", np.array([], dtype=np.int64))})


def open_failure(directory):
    """停止診断のDatasetを開く。属性step・time・errorに停止の情報を持つ。

    Parameters
    ----------
    directory : str or os.PathLike
        run_simulation()の出力フォルダ。

    Returns
    -------
    xarray.Dataset
        状態の全数値配列。中心は(r,z)、面は(r_face,z)・(r,z_face)。
    """
    return xr.open_zarr(Path(directory) / FAILURE_STORE, **_ZARR)
