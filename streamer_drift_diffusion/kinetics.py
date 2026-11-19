"""構造化された化学量論を粒子数密度の格子計算に使う。"""
from pathlib import Path

import numpy as np
from ruamel.yaml import YAML, YAMLError


MECHANISM_PATH = Path(__file__).with_name("data") / "air_chemistry.yaml"
DRY_AIR_MECHANISM_PATH = MECHANISM_PATH.with_name("dry_air_chemistry.yaml")
SPECIES_FIELDS = {
    "Electron": "ne", "N2": "n2", "O2": "o2", "N2+": "n2p", "O2+": "o2p",
    "O-": "om", "O2-": "o2m", "O": "o",
}
DRY_AIR_SPECIES_FIELDS = {**SPECIES_FIELDS, "N2O": "n2o"}
DENSITY_NAMES = tuple(DRY_AIR_SPECIES_FIELDS.values())
REACTION_FIELDS = {
    "n2-ionization": "k1", "o2-ionization": "k1_o2",
    "o2-dissociative-attachment": "k2", "o2-three-body-attachment": "k3",
}


def _nonzero_terms(row):
    """行にある非ゼロの整数係数を位置と値の組にする。

    Parameters
    ----------
    row : array_like
        整数値の反応次数または化学量論係数の一次元列。

    Returns
    -------
    tuple of tuple
        非ゼロ項の添字と係数。

    Examples
    --------
    >>> _nonzero_terms([0, 2, -1])
    ((1, 2), (2, -1))
    """
    return tuple((i, int(power)) for i, power in enumerate(row) if power)


def _mass_action(coefficient, densities, factors):
    """係数に速度式で指定した化学種の密度因子を掛ける。

    Parameters
    ----------
    coefficient : ndarray
        セルごとの反応係数、または時間刻みを掛けた係数。
    densities : tuple of ndarray
        化学種ごとのセル数密度 [m^-3]。
    factors : tuple of tuple
        化学種の添字と密度のべき指数。

    Returns
    -------
    ndarray
        各セルの係数と指定した密度因子の積。

    Examples
    --------
    >>> _mass_action(np.array([1.]), (np.array([2.]), np.array([3.])), ((0, 1), (1, 2)))
    array([18.])
    """
    for index, power in factors:
        density = densities[index]
        coefficient = coefficient * (density if power == 1 else density**power)
    return coefficient


class MassActionMechanism:
    """係数表の化学量論を、配列で評価できる質量作用則へ変換する。

    Parameters
    ----------
    data : dict
        反応定義から抽出したspecies、reaction_ids、orders、net。
        行は反応、列は化学種。ordersは速度式の次数、netは正味の生成数。
    tabulated : bool, optional
        形式4の係数表を使う。既定は形式3の旧機構。

    Raises
    ------
    ValueError
        格子で対応する化学種・反応と一致しないか、次数・化学量論が不正。

    Notes
    -----
    係数は粒子数基準のSI値を使う。反応の進行量と刻み制約で
    同じ反応次数を使い、刻み制約では対象種の密度因子を一つ除く。

    Examples
    --------
    係数表を読み込むと、このクラスが反応機構を準備する::

        table = load_reaction_table("air_mechanism_rates.json", config)
        mechanism = table.mechanism
    """

    def __init__(self, data, *, tabulated=False):
        species = DRY_AIR_SPECIES_FIELDS if tabulated else SPECIES_FIELDS
        additional_species = data.get("additional_species") or {}
        if additional_species:
            fields = [entry[0] for entry in additional_species.values()]
            if (not tabulated or species.keys() & additional_species.keys()
                    or set(species.values()) & set(fields) or len(set(fields)) != len(fields)
                    or any(not isinstance(field, str) or not field.isidentifier() for field in fields)
                    or any(len(entry) != 4 or any(type(value) is not int for value in entry[1:])
                           or min(entry[1:3]) < 0 for entry in additional_species.values())):
                raise ValueError("additional species require a tabulated mechanism, unique names "
                                 "and [field, N atoms, O atoms, charge] entries")
            species = {**species, **{name: entry[0] for name, entry in additional_species.items()}}
        self.species_fields = dict(species)
        ids = data["reaction_ids"]
        if (data["species"] != list(species) or not ids
                or any(not isinstance(name, str) or not name for name in ids)
                or len(set(ids)) != len(ids)
                or (not tabulated and ids != list(REACTION_FIELDS))):
            raise ValueError("unsupported chemistry mechanism species or reaction IDs")
        self.density_names = tuple(species.values())
        self.tabulated = tabulated
        orders = np.asarray(data["orders"], dtype=float)
        net = np.asarray(data["net"], dtype=float)
        shape = (len(ids), len(species))
        if (orders.shape != shape or net.shape != shape
                or not np.all(np.isfinite(orders)) or not np.all(np.isfinite(net))
                or np.any(orders != np.floor(orders)) or np.any(net != np.floor(net))
                or np.any(orders < 0.) or np.any(net < -orders)
                or np.any(~np.any(net, axis=1))
                or np.any(orders.sum(axis=1) < 1.)
                or (not tabulated and (np.any(orders.sum(axis=1) != (2, 2, 2, 3))
                    or np.any(orders[:, 0] != 1.) or np.any(net[:, 0] == 0.)))):
            raise ValueError("invalid chemistry mechanism orders or stoichiometry")
        # 行ごとにN原子、O原子と電荷を保存する。
        conserved = ((0, 2, 0, 2, 0, 0, 0, 0),
                     (0, 0, 2, 0, 2, 1, 2, 1),
                     (-1, 0, 0, 1, 1, -1, -1, 0))
        atoms_and_charge = np.asarray(conserved)
        if tabulated:
            atoms_and_charge = np.column_stack((atoms_and_charge, (2, 1, 0)))
        if additional_species:
            atoms_and_charge = np.column_stack(
                (atoms_and_charge, np.asarray([entry[1:] for entry in additional_species.values()]).T)
            )
        if np.any(net @ atoms_and_charge.T != 0.):
            raise ValueError("chemistry mechanism does not conserve N, O, or charge")
        self.charges = {field: int(charge) for field, charge in zip(self.density_names, atoms_and_charge[2])
                        if charge}
        self._factors = tuple(_nonzero_terms(row) for row in orders)
        self._changes = tuple(_nonzero_terms(column) for column in net.T)
        self._photoionizing = ()
        self.analytic_rates = ()
        if tabulated:
            photoionizing = np.asarray(data["photoionizing"])
            if (photoionizing.shape != (len(ids),) or photoionizing.dtype.kind != "b"
                    or np.any(photoionizing & ((orders[:, 0] != 1) | (net[:, 0] <= 0)))):
                raise ValueError("invalid photoionizing reaction flags")
            self._photoionizing = tuple(zip(np.flatnonzero(photoionizing), net[photoionizing, 0]))
            self.analytic_rates = tuple(
                (i, _exp_v1_parameters(parameters))
                for i, parameters in enumerate(data["exp_v1"]) if parameters is not None
            )
            if len(data["exp_v1"]) != len(ids):
                raise ValueError("invalid number of analytic rate definitions")
            if any(orders[i, 0] for i, _ in self.analytic_rates):
                raise ValueError("electron-impact reactions require tabulated coefficients")
        self._loss_factors = []
        for species, changes in enumerate(self._changes):
            terms = []
            for reaction, change in changes:
                if change < 0 or (species == 0 and orders[reaction, species] > 0):
                    row = orders[reaction].copy()
                    row[species] -= 1
                    terms.append((reaction, abs(change), _nonzero_terms(row)))
            if terms:
                self._loss_factors.append(tuple(terms))

    def increments(self, state, dt):
        """更新前の密度から全化学種の反応増分を評価する。

        Parameters
        ----------
        state : SimulationState
            反応係数と数密度配列。
        dt : float
            更新幅 [s]。

        Returns
        -------
        tuple of ndarray or float
            化学種ごとの数密度増分 [m^-3]。順序はDENSITY_NAMES。
            反応に関与しない種の増分は0。

        Examples
        --------
        読み込んだ機構で、現在の格子状態から電子密度の増分を評価する::

            delta_ne, *_ = state.chemistry.mechanism.increments(state, config.dt)
        """
        densities = tuple(getattr(state, name) for name in self.density_names)
        extents = tuple(
            _mass_action(dt * rate, densities, factors)
            for rate, factors in zip(self._rates(state), self._factors)
        )
        return tuple(sum((extents[i] if nu == 1 else nu * extents[i] for i, nu in terms), 0.)
                     for terms in self._changes)

    def maximum_frequency(self, state):
        """化学種ごとの変化頻度の最大を気体セルで評価する。

        Parameters
        ----------
        state : SimulationState
            反応係数、数密度、気体セルマスク。

        Returns
        -------
        float
            電子の生成・消費と他の種の消費に対する最大頻度 [s^-1]。

        Examples
        --------
        読み込んだ機構で、現在の格子状態から最大頻度を求める::

            frequency = state.chemistry.mechanism.maximum_frequency(state)
        """
        densities = tuple(getattr(state, name) for name in self.density_names)
        rates = self._rates(state)
        return max(
            float(np.max(sum(_mass_action(rates[i] if nu == 1 else nu * rates[i], densities, factors)
                             for i, nu, factors in terms)[state.gas_mask]))
            for terms in self._loss_factors
        )

    def _rates(self, state):
        if self.tabulated:
            return state.reaction_rates
        return tuple(getattr(state, name) for name in REACTION_FIELDS.values())

    def impact_ionization(self, state):
        """光電離源に指定された衝突電離の電子生成率を返す。"""
        densities = tuple(getattr(state, name) for name in self.density_names)
        return sum((weight * _mass_action(state.reaction_rates[i], densities, self._factors[i])
                    for i, weight in self._photoionizing), np.zeros_like(state.ne))


def _exp_v1_parameters(values):
    c1, c2, c3 = map(float, values)
    if not np.all(np.isfinite((c1, c2, c3))) or min(c1, c2) < 0. or c3 <= 0.:
        raise ValueError("invalid exp_v1 coefficients")
    return c1, c2, c3


def exp_v1(field, parameters):
    """換算電界[Td]でAfivo型の脱離係数を評価する。"""
    c1, c2, c3 = parameters
    return c1 * np.exp(-(c2 / (c3 + field))**2)


def load_mechanism_definition(path):
    """YAMLの反応式と係数指定を読み、反応次数・化学量論に変換する。

    Parameters
    ----------
    path : str or os.PathLike
        反応機構のYAML。``reactions``の各要素はid・reactants・products・rateを持つ。
        任意の``species``節で、基本9種（DRY_AIR_SPECIES_FIELDS）以外の化学種を
        ``名前: {field: 状態の配列名, N: N原子数, O: O原子数, charge: 電荷}``で追加できる。

    Returns
    -------
    contents : bytes
        ファイルの内容（ハッシュの記録用）。
    reactions : list of dict
        YAMLの反応定義。
    data : dict
        MassActionMechanismへ渡す化学種・反応ID・次数・正味の化学量論と、
        additional_species（追加種）・photoionizing・exp_v1。

    Raises
    ------
    ValueError
        未知の化学種、不正な係数、または原子・電荷が保存されない反応。

    Examples
    --------
    >>> _, reactions, data = load_mechanism_definition(DRY_AIR_MECHANISM_PATH)
    >>> data['species'][:3], len(data['reaction_ids']) == len(reactions)
    (['Electron', 'N2', 'O2'], True)
    """
    contents = Path(path).read_bytes()
    try:
        document = YAML(typ="safe").load(contents)
        reactions = document["reactions"]
        if any(not isinstance(reaction["rate"], dict) for reaction in reactions):
            raise ValueError("each reaction requires a structured rate definition")
        additional = {
            name: [entry["field"], entry.get("N", 0), entry.get("O", 0), entry.get("charge", 0)]
            for name, entry in (document.get("species") or {}).items()
        }
        species = list(DRY_AIR_SPECIES_FIELDS) + list(additional)
        reactants = [reaction["reactants"] for reaction in reactions]
        products = [reaction["products"] for reaction in reactions]
        for side in (*reactants, *products):
            if (not isinstance(side, dict) or any(
                    name not in species or type(count) is not int or count <= 0
                    for name, count in side.items())):
                raise ValueError("invalid species or count in reaction definition")
        data = {
            "species": species,
            "reaction_ids": [reaction["id"] for reaction in reactions],
            "orders": [[side.get(name, 0) for name in species] for side in reactants],
            "net": [[right.get(name, 0) - left.get(name, 0) for name in species]
                    for left, right in zip(reactants, products)],
            "photoionizing": [reaction.get("photoionizing", False) for reaction in reactions],
            "exp_v1": [reaction["rate"].get("exp_v1") for reaction in reactions],
        }
        if additional:
            data["additional_species"] = additional
        MassActionMechanism(data, tabulated=True)
        return contents, reactions, data
    except (KeyError, TypeError, YAMLError) as exc:
        raise ValueError(f"invalid reaction definition: {path}") from exc


def export_mechanism(gas):
    """Cantera相から対応する反応と格子用の化学量論を取り出す。

    Parameters
    ----------
    gas : cantera.Solution
        同梱のYAMLから読み込んだ反応相。

    Returns
    -------
    reactions : tuple of cantera.Reaction
        係数列と同じID順に並べた反応。
    data : dict
        JSONに保存できる化学種、反応ID、速度式の次数、正味の化学量論。

    Raises
    ------
    ValueError
        対応外の反応則、第三体、化学種、反応IDを含む。

    Examples
    --------
    Canteraが使える環境で同梱の反応機構を確認する::

        gas = cantera.Solution(str(MECHANISM_PATH))
        reactions, data = export_mechanism(gas)
    """
    reactions = {reaction.ID: reaction for reaction in gas.reactions()}
    if (set(reactions) != set(REACTION_FIELDS) or gas.n_reactions != len(REACTION_FIELDS)
            or set(gas.species_names) != set(SPECIES_FIELDS)):
        raise ValueError("unsupported Cantera mechanism species or reaction IDs")
    orders, net = [], []
    for reaction in (reactions[name] for name in REACTION_FIELDS):
        if reaction.reversible or reaction.orders or reaction.rate.type != "Arrhenius":
            raise ValueError(f"unsupported reaction law for {reaction.ID}")
        reactants = reaction.reactants
        products = reaction.products
        net.append([products.get(name, 0.) - reactants.get(name, 0.) for name in SPECIES_FIELDS])
        collider = reaction.third_body
        if collider is not None:
            if (collider.name not in SPECIES_FIELDS or collider.default_efficiency != 0.
                    or collider.efficiencies != {collider.name: 1.}):
                raise ValueError(f"unsupported third-body efficiencies for {reaction.ID}")
            reactants[collider.name] = reactants.get(collider.name, 0.) + 1.
        orders.append([reactants.get(name, 0.) for name in SPECIES_FIELDS])
    data = {"species": list(SPECIES_FIELDS), "reaction_ids": list(REACTION_FIELDS),
            "orders": orders, "net": net}
    MassActionMechanism(data)
    return tuple(reactions[name] for name in REACTION_FIELDS), data


ANALYTIC_RATE_KINDS = ("constant", "arrhenius", "electron_temperature", "field")


def analytic_rate(spec, field, *, temperature, mean_energy_ev):
    """E/N・気体温度・電子の平均エネルギーから、解析式の速度係数 [SI] を評価する。

    Parameters
    ----------
    spec : dict
        次のどれか1つのキーを持つ速度式。係数は粒子数基準のSI（二体 m³/s、三体 m⁶/s）。

        - ``constant: k``
        - ``arrhenius: {A, b, Ea}``: k = A·T^b·exp(−Ea/T)。Tは気体温度 [K]、Eaは [K]。
        - ``electron_temperature: {A, b}``: k = A·Te^b。Te = (2/3)⟨ε⟩/k_B [K]。
        - ``field: {A, B, C, D}``: k = A·exp(−(B/(C+E/N))²)·exp(−(E/N/D)²)。B, C, D は Td。
          Bを省略すると第1因子は1、Dを省略すると第2因子は1。
    field : array_like
        換算電場 E/N [Td]。
    temperature : float
        気体温度 [K]。
    mean_energy_ev : array_like
        fieldと同じ形の電子の平均エネルギー [eV]。

    Returns
    -------
    ndarray
        fieldと同じ形の速度係数。

    Raises
    ------
    ValueError
        対応外のキー、または非有限・負の係数。

    Examples
    --------
    >>> float(analytic_rate({'constant': 4e-18}, 100., temperature=300., mean_energy_ev=1.))
    4e-18
    >>> k = analytic_rate({'electron_temperature': {'A': 4.85e-12, 'b': -0.5}}, 0.,
    ...                   temperature=300., mean_energy_ev=0.0388)
    >>> round(float(k) / 2.8e-13, 2)
    1.0
    >>> float(analytic_rate({'field': {'A': 1.1e-42, 'D': 65.}}, 0., temperature=300., mean_energy_ev=1.))
    1.1e-42
    """
    if len(spec) != 1 or next(iter(spec)) not in ANALYTIC_RATE_KINDS:
        raise ValueError(f"rate must be exactly one of {ANALYTIC_RATE_KINDS}: {spec}")
    kind, value = next(iter(spec.items()))
    field = np.asarray(field, dtype=float)
    if kind == "constant":
        rate = np.full_like(field, float(value))
    elif kind == "arrhenius":
        rate = np.full_like(field, float(value["A"]) * temperature**float(value.get("b", 0.))
                            * np.exp(-float(value.get("Ea", 0.)) / temperature))
    elif kind == "electron_temperature":
        from scipy.constants import Boltzmann, elementary_charge
        electron_temperature = 2. / 3. * np.asarray(mean_energy_ev, dtype=float) * elementary_charge / Boltzmann
        rate = float(value["A"]) * electron_temperature**float(value.get("b", 0.))
    else:
        rate = np.full_like(field, float(value["A"]))
        if "B" in value:
            rate = rate * np.exp(-(float(value["B"]) / (float(value.get("C", 0.)) + field))**2)
        if "D" in value:
            rate = rate * np.exp(-(field / float(value["D"]))**2)
    if not np.all(np.isfinite(rate)) or np.any(rate < 0.):
        raise ValueError(f"invalid {kind} rate coefficients: {value}")
    return rate
