"""Create a GIF of the absolute axial reduced electric field from Ey*.csv."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from streamer_drift_diffusion.output import load_output_times
STEP_PATTERN = re.compile(r"^Ey(\d+)\.csv$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Animate the absolute axial reduced field |Ez|/N from Ey*.csv."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=SCRIPT_DIR / "outputdata",
        help="Directory containing rh.csv, zh.csv, and Ey*.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "abs_ez_progress.gif",
        help="Output GIF path",
    )
    parser.add_argument("--fps", type=float, default=5.0, help="GIF frames per second")
    parser.add_argument("--dpi", type=int, default=120, help="GIF resolution")
    parser.add_argument(
        "--vmax",
        type=float,
        default=500.0,
        help="Upper color limit in Td; use 0 for automatic global maximum",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0e-12,
        help="Simulation time step in seconds",
    )
    return parser.parse_args()


def discover_steps(input_dir: Path) -> list[int]:
    steps = []
    for path in input_dir.glob("Ey*.csv"):
        match = STEP_PATTERN.match(path.name)
        if match:
            steps.append(int(match.group(1)))
    return sorted(steps)


def load_frames(input_dir: Path) -> tuple[np.ndarray, np.ndarray, list[int], list[np.ndarray]]:
    r_m = np.atleast_1d(np.loadtxt(input_dir / "rh.csv"))
    z_m = np.atleast_1d(np.loadtxt(input_dir / "zh.csv"))
    nr, nz = len(r_m), len(z_m)
    expected = nr * nz

    valid_steps: list[int] = []
    frames: list[np.ndarray] = []
    for step in discover_steps(input_dir):
        path = input_dir / f"Ey{step}.csv"
        values = np.atleast_1d(np.loadtxt(path))
        if values.size != expected:
            print(
                f"Skipping {path.name}: {values.size} values found; "
                f"expected {expected}. The simulation may still be writing it."
            )
            continue
        # CSV order is C-compatible for(i) for(j): reshape to (r, z), then transpose.
        signed_ez = values.reshape((nr, nz), order="C")
        frames.append(np.abs(signed_ez).T)
        valid_steps.append(step)

    if not frames:
        raise FileNotFoundError(f"No complete Ey*.csv frames found in {input_dir.resolve()}")
    return r_m, z_m, valid_steps, frames


def create_gif(
    input_dir: Path,
    output_path: Path,
    fps: float = 5.0,
    dpi: int = 120,
    vmax: float = 500.0,
    dt_seconds: float = 1.0e-12,
) -> Path:
    if fps <= 0:
        raise ValueError("fps must be positive")
    if dpi <= 0:
        raise ValueError("dpi must be positive")

    r_m, z_m, steps, frames = load_frames(input_dir)
    times_s = load_output_times(input_dir, steps, dt_seconds)
    color_max = vmax if vmax > 0 else max(float(frame.max()) for frame in frames)
    if color_max <= 0:
        color_max = 1.0

    extent = (
        float(r_m[0] * 1.0e3),
        float(r_m[-1] * 1.0e3),
        float(z_m[0] * 1.0e3),
        float(z_m[-1] * 1.0e3),
    )

    plt.style.use('classic')
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 20
    fig, ax = plt.subplots(figsize=(6, 12), constrained_layout=True, facecolor="white")
    image = ax.imshow(
        frames[0],
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="turbo",
        vmin=0.0,
        vmax=color_max,
        interpolation="nearest",
    )
    # ax.set_title(r"Absolute axial reduced electric field $|E_z|/N$")
    ax.set_xlabel(r"Radial position $r$ (mm)")
    ax.set_ylabel(r"Axial position $z$ (mm)")
    ax.set_xlim(extent[0], 0.75)
    fig.colorbar(image, ax=ax, shrink=0.88, label="Reduced electric field (Td)", extend='both')
    time_text = fig.suptitle("")

    def update(frame_index: int):
        step = steps[frame_index]
        time_ns = times_s[frame_index] * 1.0e9
        image.set_data(frames[frame_index])
        time_text.set_text(f"step = {step}    time = {time_ns:.4f} ns")
        return image, time_text

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=1000.0 / fps,
        blit=False,
        repeat=True,
    )
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)

    print(f"Frames: {len(frames)} (step {steps[0]} to {steps[-1]})")
    print(f"Color range: 0 to {color_max:g} Td")
    print(f"Saved: {output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    create_gif(
        input_dir=args.input,
        output_path=args.output,
        fps=args.fps,
        dpi=args.dpi,
        vmax=args.vmax,
        dt_seconds=args.dt,
    )


if __name__ == "__main__":
    main()
