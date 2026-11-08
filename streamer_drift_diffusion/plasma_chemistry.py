"""Py3BOLOSで電子輸送係数と構造化機構の反応係数表を作る。"""
import argparse
from dataclasses import dataclass
import hashlib
from importlib.metadata import version
from io import StringIO
import json
from pathlib import Path

import numpy as np

from .kinetics import (DRY_AIR_MECHANISM_PATH, MassActionMechanism, exp_v1,
                       load_mechanism_definition)


LEGACY_EQUATIONS = (
    "Electron + N2 => 2 Electron + N2+",
    "Electron + O2 => 2 Electron + O2+",
    "Electron + O2 => O- + O",
)

@dataclass(frozen=True)
class ReactionTable:
    """一定組成・温度における電子の反応・輸送係数表。

    Parameters
    ----------
    field_td : ndarray, shape (n,)
        0から始まる、昇順の換算電場の大きさ [Td]。
    rates : ndarray
        E/Nごとの粒子数基準SI係数。形式4の列順は機構の反応ID順。
        旧形式はN₂電離、O₂電離、O₂解離付着の3列 [m³/s]。
    metadata : dict
        断面積データ、生成条件、ライブラリ版などの記録。
    swarm : ndarray, shape (n, 2), optional
        Py3BOLOSの換算移動度μN [1/(m V s)] と換算拡散係数DN [1/(m s)]。
        反応係数のみの旧形式はNone。
    mechanism : MassActionMechanism, optional
        格子計算用の化学量論。形式1・2はNone。

    Examples
    --------
    >>> table = ReactionTable(np.array([0., 100.]), np.array([[0., 0., 0.], [2., 4., 6.]]), {})
    >>> table.sample(np.array([50.])).ravel().tolist()
    [1.0, 2.0, 3.0]
    """

    field_td: np.ndarray
    rates: np.ndarray
    metadata: dict
    swarm: np.ndarray | None = None
    mechanism: MassActionMechanism | None = None

    def sample(self, field, *, column=None):
        """電子の反応係数または指定した輸送係数列を線形補間する。

        Parameters
        ----------
        field : array_like
            気体セルまたは気体に接する面の換算電場の大きさ [Td]。
        column : ndarray, shape (n,), optional
            補間する係数列。Noneなら機構の全反応係数を返す。

        Returns
        -------
        ndarray or numpy.float64
            指定列ではfieldと同じ形状で、単位はcolumnと同じ。
            反応係数では先頭軸が反応で、単位は反応次数に対応する粒子数基準のSI。

        Raises
        ------
        RuntimeError
            非有限、負、または表の範囲を超える換算電場がある。

        Examples
        --------
        >>> table = ReactionTable(np.array([0., 10.]), np.zeros((2, 3)), {})
        >>> table.sample([0., 10.]).shape
        (3, 2)
        >>> table.sample([5.], column=np.array([2., 4.])).tolist()
        [3.0]
        """
        field = np.asarray(field)
        if not np.all(np.isfinite(field)) or np.any(field < 0.):
            raise RuntimeError("electron coefficient table requires finite, nonnegative E/N [Td]")
        if np.any(field > self.field_td[-1]):
            raise RuntimeError(
                f"E/N {field.max():.6g} Td exceeds table maximum "
                f"{self.field_td[-1]:.6g} Td; regenerate the table for this range"
            )
        if column is not None:
            return np.interp(field, self.field_td, column)
        sampled = np.stack([np.interp(field, self.field_td, rate) for rate in self.rates.T])
        if self.mechanism is not None:
            for index, parameters in self.mechanism.analytic_rates:
                sampled[index] = exp_v1(field, parameters)
        return sampled


def load_reaction_table(path, config):
    """JSON係数表を読み、シミュレーションの温度・組成と照合する。

    Parameters
    ----------
    path : str or pathlib.Path
        このモジュールで生成したJSONファイル。
    config : SimulationConfig
        temperature [K]、nitrogen_fraction、oxygen_fractionを照合する。

    Returns
    -------
    ReactionTable
        軸と係数を読み取り専用にした表。

    Raises
    ------
    OSError
        ファイルを読み込めない。
    ValueError
        形式、数値、反応順、温度または組成が不正。BOLOS輸送と一定拡散を併記した。

    Examples
    --------
    生成済みの表を計算条件で読み込む::

        table = load_reaction_table("air_rates.json", SimulationConfig())
    """
    contents = Path(path).read_bytes()
    data = json.loads(contents)
    try:
        metadata = data["metadata"]
        field = np.asarray(data["field_td"], dtype=float)
        schema = data["schema"]
        if schema not in (1, 2, 3, 4) or (schema < 3 and metadata["reactions"] != list(LEGACY_EQUATIONS)):
            raise ValueError("unsupported chemistry table format or reaction order")
        tabulated = schema == 4
        rates = np.asarray(data["rates_si" if tabulated else "rates_m3_s"], dtype=float)
        mechanism = MassActionMechanism(data["mechanism"], tabulated=tabulated) if schema >= 3 else None
        columns = len(data["mechanism"]["reaction_ids"]) if tabulated else 3
        swarm = None if schema == 1 else np.asarray(
            [data["mobility_n_per_m_v_s"], data["diffusion_n_per_m_s"]], dtype=float
        ).T
        for name in ("temperature", "nitrogen_fraction", "oxygen_fraction"):
            if not np.isclose(float(metadata[name]), getattr(config, name), rtol=1e-12, atol=0.):
                raise ValueError(f"chemistry table {name} does not match the simulation")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"invalid chemistry table: {path}") from exc
    if (field.ndim != 1 or field.size < 2 or rates.shape != (field.size, columns)
            or not np.all(np.isfinite(field)) or not np.all(np.isfinite(rates))
            or field[0] != 0. or np.any(np.diff(field) <= 0.)
            or np.any(rates < 0.) or (not tabulated and np.any(rates[0] != 0.))):
        raise ValueError("chemistry table requires increasing E/N from zero and finite nonnegative rates")
    field.setflags(write=False)
    rates.setflags(write=False)
    if swarm is not None:
        if swarm.shape != (field.size, 2) or not np.all(np.isfinite(swarm)) or np.any(swarm <= 0.):
            raise ValueError("swarm table requires finite positive mobility and diffusion at every E/N")
        if config.electron_diffusion != 0.:
            raise ValueError("use electron_diffusion=0 with a Py3BOLOS transport table")
        swarm.setflags(write=False)
    metadata = {**metadata, "table_sha256": hashlib.sha256(contents).hexdigest()}
    return ReactionTable(field, rates, metadata, swarm, mechanism)


def _rate_evaluators(reactions, parsed, excluded):
    from bolos.process import Process

    evaluators, indices = [], []
    for reaction in reactions:
        spec = reaction["rate"]
        reactants = reaction["reactants"]
        if set(spec) == {"exp_v1"}:
            if sum(reactants.values()) != 2:
                raise ValueError(f"exp_v1 requires a two-body reaction for {reaction['id']}")
            parameters = tuple(map(float, spec["exp_v1"]))
            evaluators.append(lambda solver, eedf, field, parameters=parameters:
                              exp_v1(field, parameters))
            indices.append(None)
            continue
        if set(spec) != {"collision"}:
            raise ValueError(f"invalid rate source for {reaction['id']}")
        channel = spec["collision"]
        normalized = channel.get("density_normalized_cm3", False)
        threshold = channel.get("threshold_ev")
        if (type(normalized) is not bool
                or reactants != {"Electron": 1, channel["target"]: 2 if normalized else 1}):
            raise ValueError(f"collision target or rate order does not match {reaction['id']}")
        matches = [
            (index, collision) for index, collision in enumerate(parsed, 1)
            if collision["kind"] == channel["kind"] and collision["target"] == channel["target"]
            and (collision.get("product") or "").replace("^", "").replace(" ", "").upper()
            == channel["product"].upper()
            and (threshold is None or np.isclose(collision.get("threshold", 0.), threshold))
            and (index in excluded) == normalized
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one cross-section channel for {reaction['id']}, found {len(matches)}")
        index, collision = matches[0]
        process = Process(**collision)
        scale = 1e-6 if normalized else 1.
        evaluators.append(lambda solver, eedf, field, process=process, scale=scale:
                          scale * solver.rate(eedf, process))
        indices.append(index)
    return evaluators, indices


def generate_reaction_table(cross_sections, field_td, *, temperature=300., nitrogen_fraction=.8,
                            energy_max=900., energy_cells=1200,
                            mechanism_path=DRY_AIR_MECHANISM_PATH):
    """断面積データからPy3BOLOSの輸送係数と電子衝突反応係数を評価する。

    Parameters
    ----------
    cross_sections : str or pathlib.Path
        利用者が用意したBOLSIG+形式のN₂/O₂断面積ファイル。
    field_td : array_like, shape (n,)
        正で昇順の換算電場 [Td]。出力では0 Tdを先頭に加える。
    temperature : float, optional
        気体温度 [K]。既定300。
    nitrogen_fraction : float, optional
        N₂のモル分率。O₂は残りとする。既定0.8。
    energy_max : float, optional
        電子エネルギー格子の上限 [eV]。既定900。
    energy_cells : int, optional
        二次格子のセル数。既定1200。
    mechanism_path : str or pathlib.Path, optional
        反応式と係数の参照先を定義するYAML。既定は同梱の乾燥空気機構。

    Returns
    -------
    dict
        JSONへ保存できる係数表と生成条件。

    Raises
    ------
    ValueError
        設定や断面積が不正、または対象過程を一意に特定できない。
    RuntimeError
        依存関係、EEDF収束、または反応係数に異常がある。

    Notes
    -----
    YAMLから反応式を読み、次数と化学量論を係数表に保存する。
    μN、DN、電子衝突反応係数は各E/Nで求めた同じEEDFから評価する。
    密度正規化された三体付着断面積は二体衝突のEEDF計算から外し、
    得られたEEDFで積分して粒子数基準のm⁶/sへ換算する。
    脱離の解析式は係数表に保存し、格子計算時には局所E/Nで直接評価する。

    Examples
    --------
    準備した断面積ファイルから2つの電場点を計算する::

        data = generate_reaction_table("air_cross_sections.txt", [100., 300.])
    """
    try:
        from bolos import grid, parser, solver
    except ImportError as exc:
        raise RuntimeError("table generation requires keipero/Py3bolos in this environment") from exc
    from scipy.constants import Boltzmann, elementary_charge

    fields = np.asarray(field_td, dtype=float)
    if (fields.ndim != 1 or not fields.size or not np.all(np.isfinite(fields))
            or fields[0] <= 0. or np.any(np.diff(fields) <= 0.)):
        raise ValueError("generation requires strictly increasing positive E/N [Td]")
    if (not np.isfinite(temperature) or temperature <= 0. or not 0. <= nitrogen_fraction <= 1.
            or not np.isfinite(energy_max) or energy_max <= 0. or energy_cells < 2):
        raise ValueError("invalid gas composition, temperature, or energy grid")
    path = Path(cross_sections)
    contents = path.read_bytes()
    processes = []
    excluded = []
    try:
        parsed = parser.parse(StringIO(contents.decode("utf-8-sig")))
    except (ValueError, IndexError, StopIteration) as exc:
        raise ValueError(f"invalid BOLSIG+ cross-section file: {path}") from exc
    for index, collision in enumerate(parsed, 1):
        if collision["target"] not in ("N2", "O2"):
            continue
        if (collision["kind"] == "ATTACHMENT"
                and "normalized to gas density" in collision["comment"].lower()):
            excluded.append(index)
            continue
        values = np.asarray(collision["data"])
        if (values.ndim != 2 or values.shape[1] != 2 or values.shape[0] < 2
                or not np.all(np.isfinite(values)) or np.any(values < 0.)
                or np.any(np.diff(values[:, 0]) <= 0.)):
            raise ValueError(f"invalid energy or cross section in process {index}")
        processes.append((index, collision))

    mechanism_path = Path(mechanism_path)
    mechanism_contents, reactions, mechanism = load_mechanism_definition(mechanism_path)
    evaluators, indices = _rate_evaluators(reactions, parsed, excluded)
    energy_grid = grid.QuadraticGrid(0., energy_max, energy_cells)
    boltzmann = solver.BoltzmannSolver(energy_grid)
    boltzmann.load_collisions([collision for _, collision in processes])
    boltzmann.set_density("N2", nitrogen_fraction)
    boltzmann.set_density("O2", 1.-nitrogen_fraction)
    boltzmann.kT = temperature*Boltzmann/elementary_charge
    distribution = boltzmann.maxwell(boltzmann.kT)
    weights = np.sqrt(energy_grid.c)*energy_grid.d
    rates = []
    swarm = []
    max_negative = 0.
    fields = np.r_[0., fields]
    for field in fields:
        boltzmann.EN = field*solver.TOWNSEND
        boltzmann.init()
        distribution = boltzmann.converge(distribution, maxn=1000, rtol=1e-7)
        if not np.all(np.isfinite(distribution)):
            raise RuntimeError(f"nonfinite Py3BOLOS EEDF at {field:g} Td")
        positive = np.maximum(distribution, 0.)
        population = positive@weights
        negative = (np.maximum(-distribution, 0.)@weights)/population if population > 0. else np.inf
        if not np.isfinite(negative) or negative > 1e-8:
            raise RuntimeError(f"invalid Py3BOLOS EEDF at {field:g} Td (negative population {negative:g})")
        max_negative = max(max_negative, float(negative))
        transport = (boltzmann.mobility(distribution), boltzmann.diffusion(distribution))
        if not np.all(np.isfinite(transport)) or min(transport) <= 0.:
            raise RuntimeError(f"invalid Py3BOLOS mobility or diffusion at {field:g} Td")
        swarm.append(transport)
        coefficients = np.asarray([evaluate(boltzmann, positive/population, field)
                                   for evaluate in evaluators])
        if not np.all(np.isfinite(coefficients)) or np.any(coefficients < 0.):
            raise RuntimeError(f"invalid reaction rates at {field:g} Td")
        rates.append(coefficients.tolist())
        if field == 0.:
            distribution = boltzmann.maxwell(2.)
    return {
        "schema": 4, "field_td": fields.tolist(), "rates_si": rates, "mechanism": mechanism,
        "mobility_n_per_m_v_s": [float(mobility) for mobility, _ in swarm],
        "diffusion_n_per_m_s": [float(diffusion) for _, diffusion in swarm],
        "metadata": {
            "reactions": mechanism["reaction_ids"],
            "mechanism_file": mechanism_path.name,
            "mechanism_sha256": hashlib.sha256(mechanism_contents).hexdigest(),
            "temperature": temperature,
            "nitrogen_fraction": nitrogen_fraction, "oxygen_fraction": 1.-nitrogen_fraction,
            "source_file": path.name, "source_sha256": hashlib.sha256(contents).hexdigest(),
            "source_process_indices": indices, "excluded_density_normalized_processes": excluded,
            "bolos_version": version("bolos"),
            "bolos_solver_sha256": hashlib.sha256(Path(solver.__file__).read_bytes()).hexdigest(),
            "energy_max_ev": energy_max, "energy_cells": energy_cells,
            "max_negative_eedf_population": max_negative,
        },
    }


def main(argv=None):
    """コマンドラインから電子の輸送・反応係数表を生成する。

    Parameters
    ----------
    argv : list of str, optional
        引数列。Noneならコマンドラインを使用する。

    Returns
    -------
    int
        保存に成功した場合は0。

    Examples
    --------
    PowerShellで利用者の断面積ファイルを指定する::

        python -m streamer_drift_diffusion.plasma_chemistry air_cross_sections.txt air_rates.json
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cross_sections", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--temperature", type=float, default=300.)
    parser.add_argument("--nitrogen-fraction", type=float, default=.8)
    parser.add_argument("--min-td", type=float, default=.01)
    parser.add_argument("--max-td", type=float, default=10000.)
    parser.add_argument("--points", type=int, default=257)
    parser.add_argument("--energy-max", type=float, default=900.)
    parser.add_argument("--energy-cells", type=int, default=1200)
    parser.add_argument("--mechanism", type=Path, default=DRY_AIR_MECHANISM_PATH)
    args = parser.parse_args(argv)
    if not 0. < args.min_td < args.max_td < np.inf or args.points < 2:
        parser.error("require 0 < min-td < max-td and at least 2 points")
    try:
        data = generate_reaction_table(
            args.cross_sections, np.geomspace(args.min_td, args.max_td, args.points),
            temperature=args.temperature, nitrogen_fraction=args.nitrogen_fraction,
            energy_max=args.energy_max, energy_cells=args.energy_cells,
            mechanism_path=args.mechanism,
        )
        args.output.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
