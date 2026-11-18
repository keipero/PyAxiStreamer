"""python -m streamer_drift_diffusion の入口。

CLIはtyroがSimulationConfigの全項目から生成する（例: ``--nr``、``--dt-max``、
``--adaptive-timestep``／``--no-adaptive-timestep``）。``--config-file``でTOMLの設定を読み、
コマンドラインの値はそれを上書きする。項目と既定値は``--help``で確認できる。
"""
from dataclasses import dataclass, fields, replace
from pathlib import Path
import sys
import tomllib
from typing import Annotated

import tyro

from .simulation_types import SimulationConfig, PROJECT_ROOT
from .simulation import run_simulation


@dataclass(frozen=True)
class CommandLine:
    """ストリーマ放電の軸対称シミュレーションを実行する。"""

    config: Annotated[SimulationConfig, tyro.conf.OmitArgPrefixes]
    output: Path = PROJECT_ROOT / "outputdata"  # 出力フォルダ（fields.zarrのない場所）
    no_output: Annotated[bool, tyro.conf.FlagCreatePairsOff] = False  # 出力を保存しない
    quiet: Annotated[bool, tyro.conf.FlagCreatePairsOff] = False  # 進捗を表示しない
    config_file: Path | None = None  # 計算条件のTOMLファイル。コマンドラインの値が優先する


def load_config_file(path, base=None):
    """TOMLファイルの計算条件を、基準の設定へ上書きして返す。

    Parameters
    ----------
    path : str or os.PathLike
        SimulationConfigの項目名をキーとするTOML。表（[section]）は使わない。
        Noneにしたい項目（dt_max、chemistry_table）はキーを省略する。
    base : SimulationConfig, optional
        上書き前の設定。Noneなら既定値。

    Returns
    -------
    SimulationConfig
        TOMLの値で置き換えた設定。

    Raises
    ------
    ValueError
        SimulationConfigにないキーを含む場合。
    OSError, tomllib.TOMLDecodeError
        読込み・構文の失敗。

    Examples
    --------
    >>> import tempfile, os
    >>> with tempfile.TemporaryDirectory() as directory:
    ...     path = os.path.join(directory, 'run.toml')
    ...     _ = open(path, 'w', encoding='utf-8').write('nr = 12\\nadaptive_timestep = true\\n')
    ...     config = load_config_file(path)
    >>> config.nr, config.adaptive_timestep
    (12, True)
    """
    with open(path, "rb") as stream:
        values = tomllib.load(stream)
    unknown = values.keys() - {field.name for field in fields(SimulationConfig)}
    if unknown:
        raise ValueError(f"unknown SimulationConfig keys in {path}: {sorted(unknown)}")
    return replace(SimulationConfig() if base is None else base, **values)


def parse_args(argv=None):
    """コマンドライン（と指定時はTOML）からCommandLineを作る。

    Parameters
    ----------
    argv : list of str, optional
        引数列。Noneならsys.argv[1:]。

    Returns
    -------
    CommandLine
        計算条件と出力の指定。

    Raises
    ------
    SystemExit
        --help、未知のオプション、不正な値、またはTOMLの読込み失敗。

    Examples
    --------
    >>> args = parse_args(['--nr', '4', '--adaptive-timestep', '--no-output'])
    >>> args.config.nr, args.config.adaptive_timestep, args.no_output
    (4, True, True)
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    # TOMLを先に読み、その値を既定値としてコマンドラインを解釈する。
    preliminary = tyro.cli(CommandLine, args=argv, default=CommandLine(config=SimulationConfig()),
                           prog="python -m streamer_drift_diffusion")
    if preliminary.config_file is None:
        return preliminary
    try:
        base = load_config_file(preliminary.config_file)
    except (OSError, ValueError, TypeError, tomllib.TOMLDecodeError) as exc:
        print(f"error: --config-file: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    return tyro.cli(CommandLine, args=argv, default=CommandLine(config=base),
                    prog="python -m streamer_drift_diffusion")


def main(argv=None):
    """コマンドライン引数を設定へ変換し、シミュレーションを実行する。

    Parameters
    ----------
    argv : list of str, optional
        引数列。Noneならsys.argv[1:]を使用する。

    Returns
    -------
    int
        正常終了時は0。

    Raises
    ------
    SystemExit
        --help、未知のオプション、または設定・実行時エラー（終了コード2）。

    Examples
    --------
    >>> main(['--nr', '2', '--nz', '3', '--t-end', '0', '--no-output', '--quiet'])
    0
    """
    args = parse_args(argv)
    try:
        run_simulation(args.config, output_dir=args.output, write_output=not args.no_output,
                       verbose=not args.quiet)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
