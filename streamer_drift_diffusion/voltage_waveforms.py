"""2列の時間 [ns]・電圧 [kV] ファイル、線形補間、振幅・時間倍率。"""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class VoltageWaveform:
    """SI単位へ換算した電圧波形と、その最大絶対電圧を保持する。

    Attributes:
        times (np.ndarray): サンプル時刻 [s]。形状は(n_samples,)で、
            厳密な昇順に並ぶ。
        voltages (np.ndarray): timesに対応する電圧 [V]。
            timesと同じ形状の配列。
        peak_voltage (float): voltagesの絶対値の最大値 [V]。
            波形の振幅規格化に使う正の値。

    Notes:
        load_voltage_waveform()が検査済みのデータを生成する。
        このクラスを直接生成する場合は値の検査を行わない。
        frozen=Trueによりフィールドの再代入は禁止されるが、
        配列要素の変更は禁止されない。
    """

    times: np.ndarray
    voltages: np.ndarray
    peak_voltage: float


def load_voltage_waveform(path):
    """時間・電圧の2列ファイルを読み込み、SI単位の波形を返す。

    Args:
        path (str or os.PathLike): UTF-8の波形ファイルのパス。
            第1列は時刻 [ns]、第2列は電圧 [kV] とし、空白または
            タブで区切る。空行と、#以降のコメントは読み飛ばす。

    Raises:
        OSError: ファイルが存在しない、読み取り権限がないなど、
            ファイルを開くか読み込む際にエラーが発生した場合。
        ValueError: データ行が2列でない、数値に変換できない、
            換算後の値が有限でない、または時刻が厳密な昇順でない場合。
            サンプル数が2未満、または全ての電圧が0の場合にも送出する。

    Returns:
        VoltageWaveform: 時刻を[s]、電圧を[V]に換算した波形。
            peak_voltageは電圧の絶対値の最大値で、電圧の符号は保持する。
    """
    times, voltages = [], []
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.split("#", 1)[0].split()
            if not fields:
                continue
            location = f"{path}:{line_number}"
            if len(fields) != 2:
                raise ValueError(f"{location}: expected time [ns] and voltage [kV]")
            try:
                time_s, voltage_v = float(fields[0]) * 1e-9, float(fields[1]) * 1e3
            except ValueError as exc:
                raise ValueError(f"{location}: time and voltage must be numbers") from exc
            if not math.isfinite(time_s) or not math.isfinite(voltage_v):
                raise ValueError(f"{location}: time and voltage must be finite")
            if times and time_s <= times[-1]:
                raise ValueError(f"{location}: times must be strictly increasing")
            times.append(time_s)
            voltages.append(voltage_v)
    if len(times) < 2:
        raise ValueError(f"{path}: at least two samples are required")
    peak = max(map(abs, voltages))
    if peak <= 0:
        raise ValueError(f"{path}: waveform must contain a nonzero voltage")
    return VoltageWaveform(np.array(times), np.array(voltages), peak)


def validate_impulse_parameters(config):
    """インパルス波形の振幅と時間倍率が有効な値か検査する。

    Args:
        config (SimulationConfig): impulse_peak_voltage [V] と
            無次元のimpulse_time_scaleを保持する計算設定。

    Raises:
        ValueError: impulse_peak_voltageが有限かつ0以上でない場合、
            またはimpulse_time_scaleが有限かつ正でない場合。

    Returns:
        None: 検査に成功した場合、そのまま呼び出し元へ戻る。

    Notes:
        設定値の変更や波形ファイルの読み込みは行わない。
    """
    if not math.isfinite(config.impulse_peak_voltage) or config.impulse_peak_voltage < 0:
        raise ValueError("impulse_peak_voltage must be finite and nonnegative")
    if not math.isfinite(config.impulse_time_scale) or config.impulse_time_scale <= 0:
        raise ValueError("impulse_time_scale must be finite and positive")


def normalized_voltage(table, time):
    """指定時刻の電圧を線形補間し、最大絶対電圧で規格化する。

    Args:
        table (VoltageWaveform): 読み込み・検査済みの波形。
            時刻は厳密な昇順で2点以上、peak_voltageは正とする。
        time (float): 波形を評価する有限の時刻 [s]。
            applied_voltage()から呼ぶ場合は、時間倍率を適用した後の時刻。

    Returns:
        float: 補間した電圧をtable.peak_voltageで割った無次元値。
            符号を保持し、検査済みの波形では-1以上1以下となる。

    Notes:
        最初の時刻以前は先頭の値、最後の時刻以後は末尾の値を返す。
        範囲外を0にする処理や線形外挿は行わない。
        引数の妥当性はこの関数内では検査しない。
    """
    if time <= table.times[0]:
        return table.voltages[0] / table.peak_voltage
    if time >= table.times[-1]:
        return table.voltages[-1] / table.peak_voltage
    left = np.searchsorted(table.times, time, side="right") - 1
    fraction = (time - table.times[left]) / (table.times[left + 1] - table.times[left])
    left_value = table.voltages[left] / table.peak_voltage
    right_value = table.voltages[left + 1] / table.peak_voltage
    return (1 - fraction) * left_value + fraction * right_value


def applied_voltage(config, time=0.0, *, waveform=None):
    """電圧モードと指定時刻に応じて、電極の印加電圧を返す。

    Args:
        config (SimulationConfig): 電圧モードと振幅・時間倍率などの設定。
            dcではvoltage [V]、impulseではimpulse_peak_voltage [V]、
            impulse_time_scale、voltage_waveform_fileを使用する。
        time (float, optional): 印加電圧を評価する有限の時刻 [s]。
            既定値は0.0。
        waveform (VoltageWaveform, optional): 読み込み・検査済みの波形。
            キーワード引数として指定する。Noneの場合、impulseモードでは
            config.voltage_waveform_fileをこの呼び出しで読み込む。
            dcモードでは使用しない。既定値はNone。

    Raises:
        ValueError: timeが有限でない、またはvoltage_modeがdc・impulse
            以外の場合。impulseモードでは、振幅・時間倍率や読み込む
            波形データが不正な場合にも送出する。
        OSError: impulseモードで波形ファイルを開くか読み込む際に
            エラーが発生した場合。

    Returns:
        float: 印加電圧 [V]。dcではconfig.voltageをそのまま返す。
            impulseではtime / impulse_time_scaleで波形を評価し、
            最大絶対電圧で規格化した値にimpulse_peak_voltageを掛ける。

    Notes:
        時間倍率を1より大きくすると波形が時間方向に伸び、1未満では縮む。
        インパルス電圧の符号は入力波形に従う。
        評価時刻が波形範囲外なら、対応する端点の電圧を保持する。
    """
    if not math.isfinite(time):
        raise ValueError("time must be finite")
    if config.voltage_mode == "dc":
        return config.voltage
    if config.voltage_mode != "impulse":
        raise ValueError("voltage_mode must be dc or impulse")
    validate_impulse_parameters(config)
    table = load_voltage_waveform(config.voltage_waveform_file) if waveform is None else waveform
    return normalized_voltage(table, time / config.impulse_time_scale) * config.impulse_peak_voltage
