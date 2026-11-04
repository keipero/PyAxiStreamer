"""既存Python出力を、元Cおよび条件を合わせたCの新規実行と比較する。

元コード・既存CSVには書き込まない。検証用コピー、実行ログ、CSV比較表を保存する。
Usage: python tests/validate_c_outputs.py [--prepare-only] [--skip-run]
"""
import argparse
import csv
from datetime import datetime, timezone, timedelta
import difflib
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
FIELDS = ("ne", "phi", "absE", "Ex", "Ey")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_bytes(path):
    data = Path(path).read_bytes()
    return gzip.decompress(data) if str(path).endswith(".gz") else data


def csv_path(folder, filename):
    path = Path(folder) / filename
    return path if path.exists() else path.with_suffix(".csv.gz")


def saved_steps(folder):
    return sorted({int(path.name[2:].split(".", 1)[0]) for path in Path(folder).glob("ne*.csv*")})


def compress_completed(folder, *, finished=False):
    """次stepの出力開始後なら前stepの全ファイルは閉じている。既存ユーザー出力には触れない。"""
    folder = Path(folder).resolve()
    assert folder.is_relative_to(PROJECT / "validation_c_5000_20261027")
    # 前回実行のgzipファイルは今回の書き込み完了判定に使わない。
    steps = sorted(int(path.stem[2:]) for path in folder.glob("ne*.csv"))
    latest = max(steps, default=-1)
    for path in list(folder.glob("*.csv")):
        if path.name in ("rh.csv", "zh.csv"):
            eligible = finished or latest >= 100
            expected_lines = 200 if path.name == "rh.csv" else 100
        else:
            field = next((name for name in FIELDS if path.name.startswith(name)), None)
            if field is None:
                continue
            step = int(path.stem[len(field):])
            eligible = finished or step < latest
            expected_lines = 20000
        if not eligible:
            continue
        data = path.read_bytes()
        if data.count(b"\n") != expected_lines:
            if finished:
                raise ValueError(f"Incomplete CSV: {path}")
            continue
        packed = gzip.compress(data, compresslevel=6, mtime=0)
        assert gzip.decompress(packed) == data
        target = path.with_suffix(".csv.gz")
        identical = target.exists() and gzip.decompress(target.read_bytes()) == data
        # 自分で生成した再作成可能なCSVだけを対象にし、空き容量ゼロでも圧縮可能にする。
        path.unlink()
        if not identical:
            target.write_bytes(packed)


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one occurrence: {old!r}; found {text.count(old)}")
    return text.replace(old, new, 1)


def prepare(root, compiler):
    """元ソースをバイト保存し、同条件比較用の変更は別コピーにのみ適用。"""
    original = WORKSPACE / "streamer"
    metadata = {
        "created_at": datetime.now(timezone(timedelta(hours=9))).isoformat(),
        "python_output": str(PROJECT / "outputdata_notebook"),
        "source": str(original / "streamer.c"),
        "source_sha256": sha256(original / "streamer.c"),
        "python_source_sha256": {
            str(p.relative_to(PROJECT)): sha256(p)
            for p in sorted((PROJECT / "streamer_drift_diffusion").glob("*.py"))
        },
        "python_notebook_sha256": sha256(PROJECT / "run_streamer.ipynb"),
        "compiler": compiler,
        "output_storage": "New C CSVs are losslessly gzip-compressed after each completed step.",
        "compiler_version": subprocess.check_output([compiler, "--version"], text=True, errors="replace"),
        "common": {"nr": 200, "nz": 100, "delta": 1e-5, "voltage": 4000., "dt": 1e-12,
                   "max_step": 5000, "write_every": 100, "poisson_tolerance": 1e-6},
        "variants": {},
    }
    for name in ("original_c", "matched_c"):
        run = root / name
        source_dir = run / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        (run / "outputdata").mkdir(exist_ok=True)
        shutil.copy2(original / "streamer.c", source_dir / "streamer.c")
        shutil.copytree(original / "hfile", source_dir / "hfile", dirs_exist_ok=True)
        changes = []
        if name == "matched_c":
            main = source_dir / "streamer.c"
            source = main.read_bytes().decode("latin-1")
            source = replace_once(source, "bgion = 1e14;", "bgion = 1e13;")
            main.write_bytes(source.encode("latin-1"))
            changes.append("background_ionization: 1e14 -> 1e13 per update")
            memory = source_dir / "hfile" / "memory.h"
            source = memory.read_bytes().decode("latin-1")
            source = replace_once(source, "(double *)malloc((n1*(n2+1))*sizeof(double))",
                                   "(double *)calloc(n1*(n2+1),sizeof(double))")
            memory.write_bytes(source.encode("latin-1"))
            changes.append("double matrices: malloc -> calloc, including initial pphi")
            poisson = source_dir / "hfile" / "poisson.h"
            source = poisson.read_bytes().decode("latin-1")
            source = replace_once(source,
                "if (MaxPhi < fabs(phi[i][j])) MaxPhi = phi[i][j];",
                "phi[i][j] = Prev_phi + 1.5*(phi[i][j] - Prev_phi);\n"
                "\t\t\t\tif (MaxPhi < fabs(phi[i][j])) MaxPhi = fabs(phi[i][j]);")
            poisson.write_bytes(source.encode("latin-1"))
            changes.extend(["Poisson omega: 1.0 -> 1.5 (SOR)",
                            "Poisson normalization: signed phi -> abs(phi)"])
        metadata["variants"][name] = {"changes": changes,
            "source_sha256": sha256(source_dir / "streamer.c"),
            "header_sha256": {p.name: sha256(p) for p in sorted((source_dir / "hfile").glob("*.h"))}}
        patches = []
        for relative in ("streamer.c", "hfile/memory.h", "hfile/poisson.h"):
            before = (original / relative).read_bytes().decode("cp932", errors="replace").splitlines(True)
            after = (source_dir / relative).read_bytes().decode("cp932", errors="replace").splitlines(True)
            patches.extend(difflib.unified_diff(before, after, fromfile=f"original/{relative}",
                                                tofile=f"{name}/{relative}"))
        (run / "changes.diff").write_text("".join(patches), encoding="utf-8")
        command = [compiler, "-O2", "-fno-fast-math", "-ffp-contract=off", "-Wno-main-return-type",
                   "-D_CRT_SECURE_NO_WARNINGS", str(source_dir / "streamer.c"),
                   "-o", str(run / "streamer.exe")]
        metadata["variants"][name]["build_command"] = command
        temporary = root / "build_temp"
        temporary.mkdir(exist_ok=True)
        environment = dict(os.environ, TEMP=str(temporary), TMP=str(temporary))
        result = subprocess.run(command, capture_output=True, text=True, errors="replace", env=environment)
        (run / "build.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"Build failed: {run / 'build.log'}\n{result.stderr}")
        print(f"Compiled {name}", flush=True)
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def execute(root):
    """別出力先で独立に実行。実行時間は同時実行なので速度比較には使わない。"""
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    running = {}
    for name in metadata["variants"]:
        run = root / name
        output = open(run / "run.log", "wb")
        process = subprocess.Popen([str(run / "streamer.exe")], cwd=run, stdout=output,
                                   stderr=subprocess.STDOUT)
        running[name] = (process, output, time.perf_counter())
    while running:
        for name, (process, output, started) in list(running.items()):
            code = process.poll()
            current_steps = sorted(int(path.stem[2:]) for path in (root / name / "outputdata").glob("ne*.csv"))
            compress_completed(root / name / "outputdata", finished=code is not None)
            steps = saved_steps(root / name / "outputdata")
            elapsed = time.perf_counter() - started
            print(f"{name}: {'running' if code is None else 'exit='+str(code)}, "
                  f"current run step={current_steps[-1] if current_steps else 'none'}, elapsed={elapsed:.1f}s", flush=True)
            if code is not None:
                output.close()
                metadata["variants"][name]["returncode"] = code
                metadata["variants"][name]["elapsed_seconds_concurrent"] = elapsed
                metadata["variants"][name]["saved_steps"] = steps
                del running[name]
        if running:
            time.sleep(5)
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    for name, variant in metadata["variants"].items():
        if variant["returncode"]:
            raise RuntimeError(f"{name} failed, see run.log")


def finite_stats(values):
    finite = values[np.isfinite(values)]
    return dict(min=float(np.min(finite)) if finite.size else None,
                max=float(np.max(finite)) if finite.size else None,
                negative=int(np.count_nonzero(values < 0)),
                nan=int(np.count_nonzero(np.isnan(values))),
                inf=int(np.count_nonzero(np.isinf(values))))


def load_values(path):
    # WindowsのCランタイムは不正演算のNaNを nan(ind) と出力する場合がある。
    def number(value):
        if isinstance(value, bytes):
            value = value.decode("ascii")
        value = value.lower().replace("nan(ind)", "nan").replace("nan(snan)", "nan")
        return float(value)
    return np.loadtxt(path, converters=number)


def compare(root):
    python_output = PROJECT / "outputdata_notebook"
    steps = sorted(int(p.stem[2:]) for p in python_output.glob("ne*.csv"))
    rows, stats, manifest = [], [], {}
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    for step in steps:
        for field in FIELDS:
            python_path = python_output / f"{field}{step}.csv"
            actual = load_values(python_path)
            if actual.size != 20000:
                raise ValueError(f"{python_path}: expected 20000 values")
            manifest[python_path.name] = sha256(python_path)
            stats.append(dict(implementation="python", step=step, field=field, **finite_stats(actual)))
            for variant in metadata["variants"]:
                c_path = csv_path(root / variant / "outputdata", python_path.name)
                expected = load_values(c_path)
                if expected.shape != actual.shape:
                    raise ValueError(f"{c_path}: expected shape {actual.shape}, got {expected.shape}")
                stats.append(dict(implementation=variant, step=step, field=field, **finite_stats(expected)))
                finite = np.isfinite(actual) & np.isfinite(expected)
                all_finite = bool(np.all(finite))
                max_abs_error = scaled_max_error = relative_l2 = None
                if all_finite:
                    # スケール除算を先に行い、異常に大きい出力でも指標計算自体のoverflowを避ける。
                    scale = max(float(np.max(np.abs(expected))), 1e-300)
                    common_scale = max(scale, float(np.max(np.abs(actual))))
                    difference = actual / common_scale - expected / common_scale
                    scaled_max_error = float(np.max(np.abs(difference))) * (common_scale / scale)
                    max_abs_error = float(np.max(np.abs(actual - expected)))
                    relative_l2 = float(np.linalg.norm(difference) / max(np.linalg.norm(expected / scale), 1e-300)) * (common_scale / scale)
                rows.append(dict(variant=variant, step=step, field=field, all_finite=all_finite,
                    max_abs_error=max_abs_error, scaled_max_error=scaled_max_error,
                    relative_l2=relative_l2,
                    value_equal=bool(np.array_equal(actual, expected)),
                    byte_equal=python_path.read_bytes() == csv_bytes(c_path),
                    lf_normalized_equal=python_path.read_bytes().replace(b"\r\n", b"\n") == csv_bytes(c_path).replace(b"\r\n", b"\n"),
                    python_nonfinite=int(np.count_nonzero(~np.isfinite(actual))),
                    c_nonfinite=int(np.count_nonzero(~np.isfinite(expected)))))
    coordinates = []
    for name in ("rh.csv", "zh.csv"):
        p = python_output / name
        manifest[name] = sha256(p)
        for variant in metadata["variants"]:
            c = csv_path(root / variant / "outputdata", name)
            coordinates.append(dict(variant=variant, file=name,
                value_equal=bool(np.array_equal(load_values(p), load_values(c))),
                byte_equal=p.read_bytes() == csv_bytes(c),
                lf_normalized_equal=p.read_bytes().replace(b"\r\n", b"\n") == csv_bytes(c).replace(b"\r\n", b"\n")))
    for filename, data in (("comparison_by_step.csv", rows), ("field_statistics.csv", stats)):
        with open(root / filename, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    summary = dict(coordinates=coordinates, cases={})
    for variant in metadata["variants"]:
        selected = [r for r in rows if r["variant"] == variant]
        prefix = [r for r in selected if r["step"] <= 4800]
        summary["cases"][variant] = dict(files=len(selected),
            through_4800_files=len(prefix),
            through_4800_value_equal=sum(r["value_equal"] and r["all_finite"] for r in prefix),
            through_4800_lf_normalized_equal=sum(r["lf_normalized_equal"] and r["all_finite"] for r in prefix),
            value_equal=sum(r["value_equal"] for r in selected),
            finite_value_equal=sum(r["value_equal"] and r["all_finite"] for r in selected),
            byte_equal=sum(r["byte_equal"] for r in selected),
            lf_normalized_equal=sum(r["lf_normalized_equal"] for r in selected),
            finite_lf_normalized_equal=sum(r["lf_normalized_equal"] and r["all_finite"] for r in selected),
            max_scaled_error_through_4800=max(r["scaled_max_error"] for r in selected
                if r["step"] <= 4800 and r["scaled_max_error"] is not None),
            first_nonfinite_comparison_step=min((r["step"] for r in selected if not r["all_finite"]), default=None))
    summary["first_saved_negative_ne"] = {
        name: min((r["step"] for r in stats if r["implementation"] == name and r["field"] == "ne"
                   and r["negative"] > 0), default=None)
        for name in ("python", *metadata["variants"])}
    prior_rows = []
    for fresh in sorted((root / "original_c" / "outputdata").glob("*.csv*")):
        filename = fresh.name.removesuffix(".gz")
        prior = WORKSPACE / "streamer" / "outputdata" / filename
        if prior.exists():
            a, b = load_values(fresh), load_values(prior)
            prior_rows.append(dict(file=filename,
                value_equal=bool(np.array_equal(a, b)),
                lf_normalized_equal=csv_bytes(fresh).replace(b"\r\n", b"\n") == prior.read_bytes().replace(b"\r\n", b"\n")))
    if prior_rows:
        with open(root / "original_c_vs_existing_output.csv", "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(prior_rows[0]))
            writer.writeheader()
            writer.writerows(prior_rows)
        summary["original_c_vs_existing_output"] = dict(files=len(prior_rows),
            value_equal=sum(row["value_equal"] for row in prior_rows),
            lf_normalized_equal=sum(row["lf_normalized_equal"] for row in prior_rows))
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (root / "python_output_sha256.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="C:/Program Files/LLVM/bin/clang.exe")
    parser.add_argument("--output", type=Path, default=PROJECT / "validation_c_5000_20261027")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not args.skip_run:
        prepare(root, args.compiler)
        if args.prepare_only:
            return
        execute(root)
    compare(root)


if __name__ == "__main__":
    main()
