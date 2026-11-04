"""python tests/compare_julia.py [--julia=PATH]。隣接する元Julia版との回帰比較。"""
import argparse
from dataclasses import fields
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from streamer_drift_diffusion import SimulationConfig, SimulationState, run_simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--julia", default="julia")
    args = parser.parse_args()
    cases = {
        "small_dc": SimulationConfig(nr=12, nz=10, t_end=21e-12, z0=70e-6, cathode_radius=30e-6),
        "small_impulse": SimulationConfig(nr=12, nz=10, t_end=21e-12, z0=70e-6,
            cathode_radius=30e-6, voltage_mode="impulse", impulse_time_scale=2.),
        "small_negative": SimulationConfig(nr=12, nz=10, t_end=21e-12, z0=70e-6,
            cathode_radius=30e-6, voltage=-4000.),
        "standard_100": SimulationConfig(t_end=101e-12),
    }
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        subprocess.run([args.julia, "--startup-file=no", str(Path(__file__).with_name("julia_reference.jl")),
                        str(root / "julia")], check=True)
        for name, config in cases.items():
            state = run_simulation(config, output_dir=root / "python" / name, verbose=False)
            worst_scaled_error = 0.
            for field in fields(SimulationState):
                actual = getattr(state, field.name)
                if not isinstance(actual, np.ndarray):
                    continue
                expected = np.loadtxt(root / "julia" / name / f"state_{field.name}.txt").reshape(actual.shape)
                # 電荷中性の大数差では丸め誤差が増幅されるため配列尺度で正規化。
                scale = max(float(np.max(np.abs(expected))), 1e-300)
                error = float(np.max(np.abs(actual - expected))) / scale
                worst_scaled_error = max(worst_scaled_error, error)
                if not np.all(np.isfinite(actual)) or error > 1e-8:
                    raise AssertionError(f"{name}/{field.name}: max abs error / array scale = {error:.3e}")
            identical = 0
            csvs = sorted((root / "julia" / name).glob("*.csv"))
            for reference in csvs:
                actual_path = root / "python" / name / reference.name
                left, right = np.loadtxt(actual_path), np.loadtxt(reference)
                np.testing.assert_allclose(left, right, rtol=2e-6, atol=1e-20)
                identical += actual_path.read_bytes() == reference.read_bytes()
            print(f"PASS {name}: 33 arrays, max scaled error={worst_scaled_error:.3e}; "
                  f"CSV byte-identical={identical}/{len(csvs)}", flush=True)


if __name__ == "__main__":
    main()
