"""C/Julia互換: i外側・j内側、1値1行、%e (小数点以下6桁)、LF改行。"""
from pathlib import Path
import csv
from dataclasses import asdict
import json
from contextlib import ExitStack

import numpy as np
import fipy
import scipy



def write_c_values(path, values):
    """数値列をC・Julia版と互換の書式で、1値1行のCSVへ保存する。

    Parameters
    ----------
    path : str or os.PathLike
        出力ファイルのパス。
        親フォルダは呼び出し側で作成しておく。
    values : Iterable[float]
        書き込む数値の反復可能オブジェクト。
        反復で取り出される順番に保存する。

    Returns
    -------
    None
        指定したファイルへ数値を書き込む。

    Raises
    ------
    OSError
        ファイルの作成・書き込みに失敗した場合。

    Notes
    -----
    数値は小数点以下6桁の指数表記、文字コードはASCII、改行はLF。
    ヘッダーは付けず、同名ファイルがあれば上書きする。
    単位換算や入力値の検査は行わない。

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> import numpy as np
    >>> with tempfile.TemporaryDirectory() as directory:
    ...     path = Path(directory)/'values.csv'
    ...     write_c_values(path, [1., 2.])
    ...     lines = path.read_text(encoding='ascii').splitlines()
    >>> lines
    ['1.000000e+00', '2.000000e+00']
    """
    with open(path, "w", encoding="ascii", newline="\n") as stream:
        for value in values:
            stream.write(f"{value:e}\n")


def write_c_vector(path, values):
    """座標などの1次元数値配列を、配列の順番でCSVへ保存する。

    Parameters
    ----------
    path : str or os.PathLike
        出力ファイルのパス。
        親フォルダは事前に作成しておく。
    values : np.ndarray
        保存する1次元数値配列。
        計算からはセル中心座標rh・zh [m] の保存に使用する。

    Returns
    -------
    None
        指定したファイルへ配列の各要素を書き込む。

    Raises
    ------
    OSError
        ファイルの作成・書き込みに失敗した場合。

    Notes
    -----
    write_c_values()へそのまま渡し、1値1行で保存する。
    並べ替え・単位換算・配列の次元検査は行わない。

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> import numpy as np
    >>> with tempfile.TemporaryDirectory() as directory:
    ...     path = Path(directory)/'values.csv'
    ...     write_c_vector(path, np.array([1., 2.]))
    ...     lines = path.read_text(encoding='ascii').splitlines()
    >>> lines
    ['1.000000e+00', '2.000000e+00']
    """
    write_c_values(path, values)


def write_c_matrix(path, matrix):
    """2次元数値配列をC順序で1次元化し、1値1行のCSVへ保存する。

    Parameters
    ----------
    path : str or os.PathLike
        出力ファイルのパス。
        親フォルダは事前に作成しておく。
    matrix : np.ndarray
        保存する2次元数値配列。
        計算状態では形状(nr, nz)で、第1軸は半径方向、第2軸は軸方向。

    Returns
    -------
    None
        指定したファイルへ配列の各要素を書き込む。

    Raises
    ------
    OSError
        ファイルの作成・書き込みに失敗した場合。

    Notes
    -----
    np.ravel(matrix, order="C")で、半径方向iを外側、軸方向jを
    内側とする順番に並べ、write_c_values()へ渡す。
    例えば[[1, 2], [3, 4]]は1、2、3、4の順に保存する。
    入力配列は変更しない。配列形状はファイルへ保存しないため、
    読み込み時は計算のnr・nzを使って同じC順序で復元する。

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> import numpy as np
    >>> with tempfile.TemporaryDirectory() as directory:
    ...     path = Path(directory)/'values.csv'
    ...     write_c_matrix(path, np.array([[1., 2.], [3., 4.]]))
    ...     lines = path.read_text(encoding='ascii').splitlines()
    >>> lines
    ['1.000000e+00', '2.000000e+00', '3.000000e+00', '4.000000e+00']
    """
    write_c_values(path, np.ravel(matrix, order="C"))


def write_step_output(output_dir, step, state):
    """電子密度・換算電場・総電位を、指定した出力番号の5ファイルへ保存する。

    Parameters
    ----------
    output_dir : str or os.PathLike
        事前に作成済みの出力フォルダ。
    step : int
        ファイル名に付ける番号。SimulationOutputからは出力番号frameを渡す。
        受理更新数や実時刻とは別の値で、ゼロ埋めはしない。
    state : SimulationState
        ne [m⁻³]、ex/ey/abs_e [Td]、phi [V]を保持する状態。
        全てセル中心の(nr,nz)配列。

    Returns
    -------
    None
        ne、absE、Ex、Ey、phiのCSVを作成する。

    Raises
    ------
    OSError
        出力ファイルの作成または書込みが失敗。

    Notes
    -----
    CSVはC順序で1値1行、指数表記の小数点以下6桁、ASCII・LF改行。
    既存の同名ファイルは上書きする。座標・時刻表はこの関数では保存しない。
    面の電場・速度は混在させず、電場・密度を揃える処理は呼出側が担当する。

    Examples
    --------
    >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
    >>> config = SimulationConfig(nr=2, nz=3, voltage=0., nmax=0.,
    ...                           background_ionization=0., t_end=0.)
    >>> state = initialize_simulation(config)
    >>> import tempfile
    >>> from pathlib import Path
    >>> with tempfile.TemporaryDirectory() as directory:
    ...     write_step_output(directory, 0, state)
    ...     names = sorted(p.name for p in Path(directory).glob('*.csv'))
    >>> names
    ['Ex0.csv', 'Ey0.csv', 'absE0.csv', 'ne0.csv', 'phi0.csv']
    """
    for name, values in (("ne", state.ne), ("absE", state.abs_e),
                         ("Ex", state.ex), ("Ey", state.ey), ("phi", state.phi)):
        write_c_matrix(Path(output_dir) / f"{name}{step}.csv", values)


class SimulationOutput:
    """固定／適応モードの設定・実時刻・数値配列・更新履歴を保存する。

    Parameters
    ----------
    directory : str or os.PathLike
        計算結果の出力先。既存CSVを含まないフォルダ。
    config : SimulationConfig
        計算条件。単位と各設定の既定値はSimulationConfigを参照。
    state : SimulationState
        計算状態。密度・電場などは呼出側で表示時刻へ揃えておく。

    Raises
    ------
    FileExistsError
        コンテキスト開始時に出力先に既存CSVがある。
    OSError
        フォルダ・ファイルの作成や書込みに失敗。

    Attributes
    ----------
    directory : pathlib.Path
        出力先のパス。
    frame : int
        次に保存する出力番号。初期値0。
    times, history : csv.writer
        入場後に使用可能な時刻表・全受理更新の履歴ライター。
    state, config : SimulationState and SimulationConfig
        保存対象の状態・設定への参照。

    Notes
    -----
    with文で使用し、入場時に設定・座標・履歴ヘッダーを作る。
    ファイル番号は出力番号であり、受理step番号や時刻ではない。
    times.csvで出力番号・step・実時刻・直前のdtを対応付ける。

    Examples
    --------
    >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
    >>> config = SimulationConfig(nr=2, nz=3, voltage=0., nmax=0.,
    ...                           background_ionization=0., t_end=0.)
    >>> state = initialize_simulation(config)
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as directory:
    ...     with SimulationOutput(directory, config, state) as output:
    ...         output.save_frame(0, 0., 0.)
    ...     print(output.frame)
    1
    """

    def __init__(self, directory, config, state):
        """出力先・状態・設定を保持し、リソース管理を準備する。

        Parameters
        ----------
        directory : str or os.PathLike
            計算結果の出力先。既存CSVを含まないフォルダ。
        config : SimulationConfig
            計算条件。単位と各設定の既定値はSimulationConfigを参照。
        state : SimulationState
            計算状態。密度・電場などは呼出側で表示時刻へ揃えておく。

        Notes
        -----
        この時点ではファイルを作成しない。with文への入場時に作成する。

        Examples
        --------
        >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
        >>> config = SimulationConfig(nr=2, nz=3, voltage=0., nmax=0.,
        ...                           background_ionization=0., t_end=0.)
        >>> state = initialize_simulation(config)
        >>> output = SimulationOutput('output_example', config, state)
        >>> output.frame
        0
        """
        self.directory = Path(directory)
        self.config = config
        self.state = state
        self.stack = ExitStack()
        self.frame = 0

    def __enter__(self):
        """設定・座標・時刻表・受理更新の履歴を開く。

        Returns
        -------
        SimulationOutput
            入場済みの自身。

        Raises
        ------
        FileExistsError
            出力先が既存CSVを含み、条件混在を防ぐ場合。
        OSError
            ファイル・フォルダの作成や初期化に失敗。

        Notes
        -----
        初期状態の物理量フレームは自動保存しない。呼出側がprepare_step()後にsave_frame()する。
        途中のファイル初期化が失敗した場合、開いたストリームを閉じる。

        Examples
        --------
        >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
        >>> config = SimulationConfig(nr=2, nz=3, voltage=0., nmax=0.,
        ...                           background_ionization=0., t_end=0.)
        >>> state = initialize_simulation(config)
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     with SimulationOutput(directory, config, state) as output:
        ...         output.save_frame(0, 0., 0.)
        ...     print(output.frame)
        1
        """
        if self.directory.exists() and any(self.directory.glob("*.csv")):
            raise FileExistsError(f"output requires a folder without existing CSVs: {self.directory}")
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "run_config.json").write_text(
            json.dumps({"config": asdict(self.config), "numpy_version": np.__version__,
                        "poisson_backend": "fipy", "poisson_solver": type(self.state.poisson.solver).__name__,
                        "poisson_criterion": "max_abs(A*phi-b) / max_abs(b)",
                        "transport_backend": "fipy", "transport_time_scheme": "explicit Euler",
                        "fipy_version": fipy.__version__, "scipy_version": scipy.__version__,
                        "fipy_solver_suite": fipy.solvers.solver_suite,
                        "transport_boundary": "half-cell absorption sink; no inflow",
                        "model": "staggered-v1", "electrode_boundary": "staircase Dirichlet",
                        "field_layout": "cell-centered Ex/Ey/absE; complete face fields and velocities in state",
                        "field_unit": "Td",
                        "electron_diffusion_source": "constant verification coefficient; Py3bolos not coupled",
                        "outer_potential": "analytic Dirichlet voltage at outer faces",
                        "output_index": "frame", "field_and_density_times": "time_s"}, indent=2), encoding="utf-8")
        try:
            self.times_stream = self.stack.enter_context(open(self.directory / "times.csv", "w", newline="", encoding="utf-8"))
            self.history_stream = self.stack.enter_context(open(self.directory / "timestep_history.csv", "w", newline="", encoding="utf-8"))
            self.times = csv.writer(self.times_stream, lineterminator="\n")
            self.history = csv.writer(self.history_stream, lineterminator="\n")
            self.times.writerow(("frame", "step", "time_s", "dt_s", "field_time_s"))
            self.history.writerow(("step", "time_s", "dt_s", "dt_cfl_s", "dt_dielectric_s",
                                   "dt_chemistry_s", "limiting", "retries", "min_ne", "max_ne", "max_field_td",
                                   "electron_number", "charge_c", "electrode_electron_loss",
                                   "ground_electron_loss", "outer_electron_loss", "dt_diffusion_s"))
            write_c_vector(self.directory / "rh.csv", self.state.rh)
            write_c_vector(self.directory / "zh.csv", self.state.zh)
            write_c_matrix(self.directory / "gas_mask.csv", self.state.gas_mask)
        except Exception:
            self.stack.close()
            raise
        return self

    def record_step(self, step, time, dt, limits, limiting, retries):
        """受理した1更新の時間幅・制約・密度・電場・粒子収支を記録する。

        Parameters
        ----------
        step : int
            受理更新の通し番号。
        time : float
            更新後の実時刻 [s]。
        dt : float
            受理した更新幅 [s]。
        limits : TimestepLimits
            更新前状態から評価した各時間上限 [s]。
        limiting : str
            実際の支配条件。fixed、output、waveform、t_end、retryなどを含む。
        retries : int
            この更新で棄却・再試行した回数。

        Returns
        -------
        None
            timestep_history.csvに1行追加する。

        Raises
        ------
        OSError
            履歴への書込みに失敗。

        Notes
        -----
        密度・電場はstateの現在値を読む。時刻の更新や電場再計算は行わない。
        実粒子数・電荷の体積積分で2πを一度掛ける。

        Examples
        --------
        >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
        >>> config = SimulationConfig(nr=2, nz=3, voltage=0., nmax=0.,
        ...                           background_ionization=0., t_end=0.)
        >>> state = initialize_simulation(config)
        >>> from streamer_drift_diffusion.timestepping import timestep_limits, advance_density
        >>> from streamer_drift_diffusion.simulation import prepare_step
        >>> prepare_step(state, config, 0., verbose=False)
        >>> limits = timestep_limits(state, config)
        >>> dt, retries = advance_density(state, config, config.dt)
        >>> prepare_step(state, config, dt, verbose=False)
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     with SimulationOutput(directory, config, state) as output:
        ...         output.record_step(1, dt, dt, limits, 'fixed', retries)
        ...     print(len((Path(directory)/'timestep_history.csv').read_text().splitlines()))
        2
        """
        volume = self.state.volume * (2 * np.pi)
        charge_density = self.config.elementary_charge * (
            self.state.n2p + self.state.o2p - self.state.om - self.state.o2m - self.state.ne)
        self.history.writerow((step, time, dt, limits.cfl, limits.dielectric, limits.chemistry,
                               limiting, retries, float(np.min(self.state.ne[self.state.gas_mask])),
                               float(np.max(self.state.ne)), float(np.max(self.state.abs_e)),
                               float(np.sum(self.state.ne * volume)), float(np.sum(charge_density * volume)),
                               *self.state.electron_losses, limits.diffusion))

    def save_frame(self, step, time, dt):
        """同一時刻の物理量フレームと時刻表の1行を保存する。

        Parameters
        ----------
        step : int
            現在までの受理更新数。初期状態では0。
        time : float
            保存する状態の実時刻 [s]。
        dt : float
            直前に受理した幅 [s]。初期状態では0。

        Returns
        -------
        None
            5種類の物理量CSVを書き、次の出力番号へ進める。

        Raises
        ------
        OSError
            物理量や時刻表への書込み・flushが失敗。

        Notes
        -----
        呼出側が電位・電場・密度をtimeへ揃えておく。ここでは再計算しない。
        ファイル名の番号には内部のframeを使い、stepを直接使わない。

        Examples
        --------
        >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
        >>> config = SimulationConfig(nr=2, nz=3, voltage=0., nmax=0.,
        ...                           background_ionization=0., t_end=0.)
        >>> state = initialize_simulation(config)
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     with SimulationOutput(directory, config, state) as output:
        ...         output.save_frame(0, 0., 0.)
        ...     print(output.frame)
        1
        """
        write_step_output(self.directory, self.frame, self.state)
        self.times.writerow((self.frame, step, time, dt, time))
        self.frame += 1
        self.times_stream.flush()
        self.history_stream.flush()

    def save_failure(self, step, time, exc):
        """停止理由と、その時点の数値状態配列を保存する。

        Parameters
        ----------
        step : int
            停止時の受理更新数。
        time : float
            停止時の時計 [s]。
        exc : BaseException
            停止原因。文字列としてfailure.jsonへ保存する。

        Returns
        -------
        None
            failure.jsonとfailure_state.npzを作成する。

        Raises
        ------
        OSError
            停止診断ファイルの保存に失敗。

        Notes
        -----
        FiPyのメッシュ・方程式・CellVariableは保存せず、ndarray属性とne・phiの2次元値だけを含める。
        ne_var・phi_varはその名前で保存せず、従来どおりne・phiという数値配列を保存する。

        Examples
        --------
        >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
        >>> config = SimulationConfig(nr=2, nz=3, voltage=0., nmax=0.,
        ...                           background_ionization=0., t_end=0.)
        >>> state = initialize_simulation(config)
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     with SimulationOutput(directory, config, state) as output:
        ...         output.save_failure(0, 0., RuntimeError('example'))
        ...     print((Path(directory)/'failure_state.npz').exists())
        True
        """
        from dataclasses import fields
        (self.directory / "failure.json").write_text(
            json.dumps({"step": step, "time_s": time, "error": str(exc)}, indent=2), encoding="utf-8")
        np.savez_compressed(self.directory / "failure_state.npz",
                            ne=self.state.ne, phi=self.state.phi,
                            **{field.name: getattr(self.state, field.name) for field in fields(self.state)
                               if isinstance(getattr(self.state, field.name), np.ndarray)})

    def __exit__(self, *exc):
        """開いた出力ストリームを閉じ、例外を呼出側へ伝える。

        Parameters
        ----------
        *exc : object
            with文が渡す例外型・例外値・トレースバック。未使用。

        Returns
        -------
        None
            例外を抑制しない。

        Notes
        -----
        通常は直接呼ばず、with文の終了処理として使用する。

        Examples
        --------
        >>> from streamer_drift_diffusion import SimulationConfig, initialize_simulation
        >>> config = SimulationConfig(nr=2, nz=3, voltage=0., nmax=0.,
        ...                           background_ionization=0., t_end=0.)
        >>> state = initialize_simulation(config)
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     with SimulationOutput(directory, config, state) as output:
        ...         output.save_frame(0, 0., 0.)
        ...     print(output.frame)
        1
        """
        self.stack.close()


def load_output_times(directory, frames, dt_seconds=1e-12):
    """出力番号に対応する物理時刻を、時刻表または旧形式の規則から返す。

    Parameters
    ----------
    directory : str or os.PathLike
        時刻表と計算結果を置いたフォルダ。読込みのみを行う。
    frames : sequence of int
        時刻を求める出力番号。与えられた順序で返す。
    dt_seconds : float, optional
        times.csvがない旧出力用のdt [s]。既定1e-12。

    Returns
    -------
    ndarray, shape (len(frames),)
        各出力番号に対応する時刻 [s]。

    Raises
    ------
    ValueError
        旧dtが不正、時刻表の列・番号・単調性が不正、または指定番号が欠落。
    OSError
        時刻表の読込みに失敗。

    Notes
    -----
    両モードともtimes.csvを優先する。時刻表があるのに不正・欠落している場合は
    番号*dt_secondsへフォールバックしない。時刻を推測して誤表示することを防ぐ。

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> with tempfile.TemporaryDirectory() as directory:
    ...     legacy = load_output_times(directory, [0, 2], 1e-12)
    ...     _ = (Path(directory)/'times.csv').write_text('frame,time_s\\n0,0\\n1,3e-12\\n', encoding='utf-8')
    ...     actual = load_output_times(directory, [0, 1])
    >>> legacy.tolist(), actual.tolist()
    ([0.0, 2e-12], [0.0, 3e-12])
    """
    path = Path(directory) / "times.csv"
    if not path.exists():
        if not np.isfinite(dt_seconds) or dt_seconds <= 0:
            raise ValueError("dt_seconds must be finite and positive")
        return np.asarray(frames, dtype=float) * dt_seconds
    mapping = {}
    previous = -np.inf
    with path.open(encoding="utf-8", newline="") as stream:
        try:
            for row in csv.DictReader(stream):
                frame, time = int(row["frame"]), float(row["time_s"])
                if frame < 0 or frame in mapping or not np.isfinite(time) or time < 0 or time <= previous:
                    raise ValueError("duplicate frame or invalid/non-increasing time")
                mapping[frame] = time
                previous = time
            return np.array([mapping[int(frame)] for frame in frames])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid or incomplete time mapping: {path}: {exc}") from exc
