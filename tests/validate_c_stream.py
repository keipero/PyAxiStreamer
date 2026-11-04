"""CのCSVを標準出力パイプで受けて比較する。全C出力のディスク保存は不要。"""
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np

import validate_c_outputs as base

ROOT = base.PROJECT / "validation_c_5000_20261027"
WRAPPER = r'''
#include <stdio.h>
#include <stdarg.h>
static FILE *validation_fopen(const char *name, const char *mode) {
    (void)mode;
    printf("@@CSV BEGIN %s\n", name);
    return stdout;
}
static int validation_fprintf(FILE *file, const char *format, ...) {
    (void)file;
    va_list args;
    va_start(args, format);
    int result = vfprintf(stdout, format, args);
    va_end(args);
    return result;
}
static int validation_fclose(FILE *file) {
    (void)file;
    printf("@@CSV END\n");
    return fflush(stdout);
}
#define fopen validation_fopen
#define fprintf validation_fprintf
#define fclose validation_fclose
#define main original_streamer_main
#include "source/streamer.c"
#undef main
#undef fopen
#undef fprintf
#undef fclose
int main(void) {
    original_streamer_main();
    return 0;
}
'''


def metrics(actual, expected):
    all_finite = bool(np.all(np.isfinite(actual)) and np.all(np.isfinite(expected)))
    absolute = scaled = l2 = None
    if all_finite:
        scale = max(float(np.max(np.abs(expected))), 1e-300)
        common = max(scale, float(np.max(np.abs(actual))))
        difference = actual / common - expected / common
        absolute = float(np.max(np.abs(actual - expected)))
        scaled = float(np.max(np.abs(difference))) * (common / scale)
        l2 = float(np.linalg.norm(difference) / max(np.linalg.norm(expected / scale), 1e-300)) * (common / scale)
    return dict(all_finite=all_finite, max_abs_error=absolute, scaled_max_error=scaled,
                relative_l2=l2, value_equal=bool(np.array_equal(actual, expected)),
                python_nonfinite=int(np.count_nonzero(~np.isfinite(actual))),
                c_nonfinite=int(np.count_nonzero(~np.isfinite(expected))))


def run_variant(name, compiler):
    folder = ROOT / name
    wrapper = folder / "stream_wrapper.c"
    wrapper.write_text(WRAPPER, encoding="ascii")
    executable = folder / "streamer_stream.exe"
    environment = dict(os.environ, TEMP=str(ROOT / "build_temp"), TMP=str(ROOT / "build_temp"))
    command = [compiler, "-O2", "-fno-fast-math", "-ffp-contract=off", "-D_CRT_SECURE_NO_WARNINGS",
               str(wrapper), "-o", str(executable)]
    built = subprocess.run(command, env=environment, capture_output=True, text=True, errors="replace")
    (folder / "stream_build.log").write_text(built.stdout + built.stderr, encoding="utf-8")
    if built.returncode:
        raise RuntimeError(built.stderr)
    print(f"Streaming C started: {name}", flush=True)
    started = time.perf_counter()
    process = subprocess.Popen([str(executable)], cwd=folder, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    rows, stats, coordinates, prior, log = [], [], [], [], []
    python_stats, manifest = {}, {}
    filename, content = None, []
    for line in process.stdout:
        clean = line.strip()
        if clean.startswith(b"@@CSV BEGIN "):
            if filename is not None:
                raise RuntimeError("Nested CSV output")
            filename = clean[len(b"@@CSV BEGIN "):].decode("ascii").replace("\\", "/").split("/")[-1]
            content = []
        elif clean == b"@@CSV END":
            if filename is None:
                raise RuntimeError("Unexpected CSV END")
            raw = b"".join(content)
            text = raw.decode("ascii").lower().replace("nan(ind)", "nan").replace("nan(snan)", "nan")
            c_values = np.fromstring(text, sep=" ")
            python_path = base.PROJECT / "outputdata_notebook" / filename
            python_values = base.load_values(python_path)
            if c_values.shape != python_values.shape:
                raise RuntimeError(f"{name}/{filename}: shape {c_values.shape}, expected {python_values.shape}")
            python_raw = python_path.read_bytes()
            manifest[filename] = base.sha256(python_path)
            if filename in ("rh.csv", "zh.csv"):
                coordinates.append(dict(variant=name, file=filename,
                    value_equal=bool(np.array_equal(python_values, c_values)),
                    byte_equal=python_raw == raw,
                    lf_normalized_equal=python_raw.replace(b"\r\n", b"\n") == raw.replace(b"\r\n", b"\n")))
            else:
                match = re.fullmatch(r"(ne|absE|Ex|Ey|phi)(\d+)\.csv", filename)
                if match is None:
                    raise RuntimeError(filename)
                field, step = match[1], int(match[2])
                rows.append(dict(variant=name, step=step, field=field, **metrics(python_values, c_values),
                    byte_equal=python_raw == raw,
                    lf_normalized_equal=python_raw.replace(b"\r\n", b"\n") == raw.replace(b"\r\n", b"\n")))
                stats.append(dict(implementation=name, step=step, field=field, **base.finite_stats(c_values)))
                python_stats[(step, field)] = dict(implementation="python", step=step, field=field,
                                                   **base.finite_stats(python_values))
            if name == "original_c":
                prior_path = base.WORKSPACE / "streamer" / "outputdata" / filename
                if prior_path.exists():
                    prior.append(dict(file=filename,
                        value_equal=bool(np.array_equal(c_values, base.load_values(prior_path))),
                        lf_normalized_equal=raw.replace(b"\r\n", b"\n") == prior_path.read_bytes().replace(b"\r\n", b"\n")))
            filename, content = None, []
        elif filename is not None:
            content.append(line)
        else:
            message = clean.decode("utf-8", errors="replace")
            if message:
                log.append(message)
                if message.startswith("step ="):
                    print(f"{name}: {message}", flush=True)
    code = process.wait()
    elapsed = time.perf_counter() - started
    (folder / "stream_run.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    if code != 0 or filename is not None or len(rows) != 255:
        raise RuntimeError(f"{name}: returncode={code}, received={len(rows)}/255 fields, open={filename}")
    print(f"Streaming C complete: {name}, elapsed={elapsed:.1f}s", flush=True)
    return dict(name=name, rows=rows, stats=stats, python_stats=python_stats, coordinates=coordinates,
                prior=prior, manifest=manifest, command=command, elapsed=elapsed, returncode=code)


def save_csv(name, rows):
    with open(ROOT / name, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    metadata = json.loads((ROOT / "metadata.json").read_text(encoding="utf-8"))
    compiler = metadata["compiler"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda name: run_variant(name, compiler), ("original_c", "matched_c")))
    rows, stats, coordinates, prior, manifest = [], [], [], [], {}
    python_stats = {}
    for result in results:
        rows.extend(result["rows"])
        stats.extend(result["stats"])
        coordinates.extend(result["coordinates"])
        prior.extend(result["prior"])
        manifest.update(result["manifest"])
        python_stats.update(result["python_stats"])
        variant = metadata["variants"][result["name"]]
        variant.update(returncode=result["returncode"], build_command=result["command"],
                       elapsed_seconds_concurrent=result["elapsed"], saved_steps=list(range(0, 5001, 100)))
    stats.extend(python_stats.values())
    rows.sort(key=lambda row: (row["variant"], row["step"], row["field"]))
    stats.sort(key=lambda row: (row["implementation"], row["step"], row["field"]))
    summary = dict(coordinates=coordinates, cases={})
    for name in ("original_c", "matched_c"):
        selected = [row for row in rows if row["variant"] == name]
        prefix = [row for row in selected if row["step"] <= 4800]
        summary["cases"][name] = dict(files=len(selected),
            value_equal=sum(row["value_equal"] for row in selected),
            finite_value_equal=sum(row["value_equal"] and row["all_finite"] for row in selected),
            byte_equal=sum(row["byte_equal"] for row in selected),
            lf_normalized_equal=sum(row["lf_normalized_equal"] for row in selected),
            through_4800_files=len(prefix),
            through_4800_value_equal=sum(row["value_equal"] and row["all_finite"] for row in prefix),
            through_4800_lf_normalized_equal=sum(row["lf_normalized_equal"] and row["all_finite"] for row in prefix),
            max_scaled_error_through_4800=max(row["scaled_max_error"] for row in prefix if row["scaled_max_error"] is not None),
            first_nonfinite_comparison_step=min((row["step"] for row in selected if not row["all_finite"]), default=None))
    summary["first_saved_negative_ne"] = {
        name: min((row["step"] for row in stats if row["implementation"] == name and row["field"] == "ne"
                   and row["negative"] > 0), default=None) for name in ("python", "original_c", "matched_c")}
    summary["original_c_vs_existing_output"] = dict(files=len(prior),
        value_equal=sum(row["value_equal"] for row in prior),
        lf_normalized_equal=sum(row["lf_normalized_equal"] for row in prior))
    metadata["execution_mode"] = "CSV fopen/fprintf/fclose redirected to stdout by a wrapper; calculations unchanged"
    metadata["output_storage"] = "Complete 5000-step C outputs compared through pipes; only aggregate tables saved"
    metadata["disk_recovery"] = "Disk full during file-output attempts; partial gzip CSVs retained, final validation uses pipes"
    save_csv("comparison_by_step.csv", rows)
    save_csv("field_statistics.csv", stats)
    save_csv("original_c_vs_existing_output.csv", prior)
    for name, data in (("metadata.json", metadata), ("summary.json", summary), ("python_output_sha256.json", manifest)):
        (ROOT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print("Remaining disk MB:", shutil.disk_usage(ROOT).free / 1e6, flush=True)


if __name__ == "__main__":
    main()
