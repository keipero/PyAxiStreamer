"""C/Julia互換: i外側・j内側、1値1行、%e (小数点以下6桁)、LF改行。"""
from pathlib import Path
import csv
from dataclasses import asdict
import json
from contextlib import ExitStack

import numpy as np
import fipy
import scipy

from ._acceleration import NUMBA_AVAILABLE, get_num_threads


def write_c_values(path, values):
    """数値列をC・Julia版と互換の書式で、1値1行のCSVへ保存する。

    Args:
        path (str or os.PathLike): 出力ファイルのパス。
            親フォルダは呼び出し側で作成しておく。
        values (Iterable[float]): 書き込む数値の反復可能オブジェクト。
            反復で取り出される順番に保存する。

    Raises:
        OSError: ファイルの作成・書き込みに失敗した場合。

    Returns:
        None: 指定したファイルへ数値を書き込む。

    Notes:
        数値は小数点以下6桁の指数表記、文字コードはASCII、改行はLF。
        ヘッダーは付けず、同名ファイルがあれば上書きする。
        単位換算や入力値の検査は行わない。
    """
    with open(path, "w", encoding="ascii", newline="\n") as stream:
        for value in values:
            stream.write(f"{value:e}\n")


def write_c_vector(path, values):
    """座標などの1次元数値配列を、配列の順番でCSVへ保存する。

    Args:
        path (str or os.PathLike): 出力ファイルのパス。
            親フォルダは事前に作成しておく。
        values (np.ndarray): 保存する1次元数値配列。
            計算からはセル中心座標rh・zh [m] の保存に使用する。

    Raises:
        OSError: ファイルの作成・書き込みに失敗した場合。

    Returns:
        None: 指定したファイルへ配列の各要素を書き込む。

    Notes:
        write_c_values()へそのまま渡し、1値1行で保存する。
        並べ替え・単位換算・配列の次元検査は行わない。
    """
    write_c_values(path, values)


def write_c_matrix(path, matrix):
    """2次元数値配列をC順序で1次元化し、1値1行のCSVへ保存する。

    Args:
        path (str or os.PathLike): 出力ファイルのパス。
            親フォルダは事前に作成しておく。
        matrix (np.ndarray): 保存する2次元数値配列。
            計算状態では形状(nr, nz)で、第1軸は半径方向、第2軸は軸方向。

    Raises:
        OSError: ファイルの作成・書き込みに失敗した場合。

    Returns:
        None: 指定したファイルへ配列の各要素を書き込む。

    Notes:
        np.ravel(matrix, order="C")で、半径方向iを外側、軸方向jを
        内側とする順番に並べ、write_c_values()へ渡す。
        例えば[[1, 2], [3, 4]]は1、2、3、4の順に保存する。
        入力配列は変更しない。配列形状はファイルへ保存しないため、
        読み込み時は計算のnr・nzを使って同じC順序で復元する。
    """
    write_c_values(path, np.ravel(matrix, order="C"))


def write_step_output(output_dir, step, state):
    """指定stepの電子密度・換算電場・電位を5個のCSVへ保存する。

    Args:
        output_dir (str or os.PathLike): 事前に作成済みの出力フォルダ。
        step (int): ファイル名へ付けるstep番号。ゼロ埋めは行わない。
        state (SimulationState): 保存するne [m^-3]、abs_e・ex・ey [Td]、
            phi [V] を保持する計算状態。各配列の形状は(nr, nz)。

    Raises:
        OSError: 出力ファイルの作成・書き込みに失敗した場合。

    Returns:
        None: 出力フォルダへ5種類の物理量を書き込む。

    Notes:
        ne、abs_e、ex、ey、phiの順にwrite_c_matrix()を呼ぶ。
        ファイル名はne{step}.csv、absE{step}.csv、Ex{step}.csv、
        Ey{step}.csv、phi{step}.csvとする。
        Exは半径方向、Eyは軸方向のセル中心換算電場であり、単位はTd。
        absE=hypot(Ex,Ey)。内部の面電場・速度をこのCSVへ混在させない。
        同名ファイルは上書きする。座標・時刻・配列形状は保存しない。
        計算状態や保存間隔は更新せず、渡された配列をそのまま保存する。
    """
    for name, values in (("ne", state.ne), ("absE", state.abs_e),
                         ("Ex", state.ex), ("Ey", state.ey), ("phi", state.phi)):
        write_c_matrix(Path(output_dir) / f"{name}{step}.csv", values)


class AdaptiveOutput:
    """適応計算の実時刻・受理刻み・設定を保存するコンテキストマネージャ。

    CSV番号は出力番号。times.csvで受理step・実時刻・dtへ対応付ける。
    既存CSVのあるフォルダは条件混在を避けるため再利用しない。
    """

    def __init__(self, directory, config, state):
        self.directory = Path(directory)
        self.config = config
        self.state = state
        self.stack = ExitStack()
        self.frame = 0

    def __enter__(self):
        if self.directory.exists() and any(self.directory.glob("*.csv")):
            raise FileExistsError(f"adaptive output requires a folder without existing CSVs: {self.directory}")
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "run_config.json").write_text(
            json.dumps({"config": asdict(self.config), "numpy_version": np.__version__,
                        "numba_available": NUMBA_AVAILABLE, "num_threads_used": get_num_threads(),
                        "transport_backend": "fipy", "transport_time_scheme": "explicit Euler",
                        "fipy_version": fipy.__version__, "scipy_version": scipy.__version__,
                        "fipy_solver_suite": fipy.solvers.solver_suite,
                        "transport_boundary": "half-cell absorption sink; no inflow",
                        "model": "staggered-v1", "electrode_boundary": "staircase Dirichlet",
                        "field_layout": "cell-centered Ex/Ey/absE; complete face fields and velocities in state",
                        "field_unit": "Td",
                        "electron_diffusion_source": "constant verification coefficient; Py3bolos not coupled",
                        "outer_potential": "analytic background, zero space-charge correction",
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
        """受理した全更新の刻み・制約・密度と電場の範囲を保存する。"""
        volume = self.state.volume * (2 * np.pi)
        charge_density = self.config.elementary_charge * (
            self.state.n2p + self.state.o2p - self.state.om - self.state.o2m - self.state.ne)
        self.history.writerow((step, time, dt, limits.cfl, limits.dielectric, limits.chemistry,
                               limiting, retries, float(np.min(self.state.ne[self.state.gas_mask])),
                               float(np.max(self.state.ne)), float(np.max(self.state.abs_e)),
                               float(np.sum(self.state.ne * volume)), float(np.sum(charge_density * volume)),
                               *self.state.electron_losses, limits.diffusion))

    def save_frame(self, step, time, dt):
        """同一時刻の密度と再計算済みの電場を保存し、対応表へ記録する。"""
        write_step_output(self.directory, self.frame, self.state)
        self.times.writerow((self.frame, step, time, dt, time))
        self.frame += 1
        self.times_stream.flush()
        self.history_stream.flush()

    def save_failure(self, step, time, exc):
        """停止理由と停止時の状態配列を、再調査用に保存する。"""
        from dataclasses import fields
        (self.directory / "failure.json").write_text(
            json.dumps({"step": step, "time_s": time, "error": str(exc)}, indent=2), encoding="utf-8")
        np.savez_compressed(self.directory / "failure_state.npz",
                            **{field.name: getattr(self.state, field.name) for field in fields(self.state)
                               if isinstance(getattr(self.state, field.name), np.ndarray)})

    def __exit__(self, *exc):
        self.stack.close()


def load_output_times(directory, frames, dt_seconds=1e-12):
    """CSV番号に対応する時刻[s]を返す。適応出力ではtimes.csvを優先する。

    従来出力にtimes.csvがなければ、番号*dt_secondsへフォールバックする。
    対応表が存在するのに不正・欠落がある場合はValueErrorとし、
    誤った固定刻みの時刻を推測しない。
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
