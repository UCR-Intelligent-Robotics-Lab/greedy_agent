import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def trim_after_cpu_drop(data: Dict[str, np.ndarray], threshold: float = 100.0) -> Optional[Dict[str, np.ndarray]]:
    """Trim data arrays after CPU utilisation drops below the given threshold."""

    cpu_total = data["cpu_total"]
    below_threshold = np.where(cpu_total < threshold)[0]
    if below_threshold.size == 0:
        return data

    cutoff = below_threshold[0]
    if cutoff <= 0:
        return None

    indices = np.arange(cutoff)
    trimmed_data = {
        "elapsed": data["elapsed"][indices],
        "cpu_total": cpu_total[indices],
        "gpu_util": data["gpu_util"][indices],
        "temperatures": {key: values[indices] for key, values in data["temperatures"].items()},
    }

    # Rebase elapsed time to start at zero after trimming
    trimmed_data["elapsed"] = trimmed_data["elapsed"] - trimmed_data["elapsed"][0]

    return trimmed_data if trimmed_data["elapsed"].size > 0 else None


def parse_profile_log(log_path: Path, sensors: Optional[List[str]] = None) -> Optional[Dict[str, np.ndarray]]:
    """Parse a Jetson profile log into arrays for plotting."""

    if sensors is None:
        sensors = ["TJ"]

    sensor_keys = [sensor.upper() for sensor in sensors]
    elapsed_seconds: List[float] = []
    cpu_total: List[float] = []
    gpu_util: List[float] = []
    temp_data: Dict[str, List[float]] = {key: [] for key in sensor_keys}

    timestamp0: Optional[datetime] = None

    with log_path.open("r", encoding="utf-8", errors="ignore") as log_file:
        for raw_line in log_file:
            line = raw_line.strip()
            if not line:
                continue

            # Timestamp parsing (assumes format: MM-DD-YYYY HH:MM:SS ...)
            try:
                timestamp = datetime.strptime(line[:19], "%m-%d-%Y %H:%M:%S")
            except ValueError:
                continue

            if timestamp0 is None:
                timestamp0 = timestamp
            elapsed_seconds.append((timestamp - timestamp0).total_seconds())

            # CPU utilisation (sum across cores)
            cpu_match = re.search(r"CPU \[(.*?)\]", line)
            if cpu_match:
                core_samples = re.findall(r"(\d+(?:\.\d+)?)%", cpu_match.group(1))
                if core_samples:
                    values = [float(sample) for sample in core_samples]
                    cpu_total.append(sum(values))
                else:
                    cpu_total.append(np.nan)
            else:
                cpu_total.append(np.nan)

            # GPU utilisation
            gpu_match = re.search(r"GR3D_FREQ (\d+(?:\.\d+)?)%", line)
            gpu_util.append(float(gpu_match.group(1)) if gpu_match else np.nan)

            # Temperatures
            temp_matches = re.findall(r"([A-Za-z0-9]+)@(-?\d+(?:\.\d+)?)C", line)
            temp_map = {name.upper(): float(value) for name, value in temp_matches}
            for key in sensor_keys:
                temp_data[key].append(temp_map.get(key, np.nan))

    if not elapsed_seconds:
        return None

    return {
        "elapsed": np.asarray(elapsed_seconds),
        "cpu_total": np.asarray(cpu_total, dtype=float),
        "gpu_util": np.asarray(gpu_util, dtype=float),
        "temperatures": {key: np.asarray(values, dtype=float) for key, values in temp_data.items()},
    }


def plot_profile(log_path: Path, data: Dict[str, np.ndarray], output_dir: Path, show: bool = False) -> None:
    elapsed = data["elapsed"] / 60.0  # minutes for readability

    cpu_total = data["cpu_total"]
    gpu_util = data["gpu_util"]

    temperatures = data["temperatures"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    title = log_path.name
    print(title)
    if title.startswith('autorun_'):
        title = title[len('autorun_'):]
    if title.endswith('.txt'):
        title = title[:-len('.txt')]
    print(title)
    # fig.suptitle(title)
    fig.suptitle("")

    axes[0].plot(elapsed, cpu_total, label="CPU util (%)", color="#1f77b4")
    axes[0].set_ylabel("CPU util. (%)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(elapsed, gpu_util, label="GPU util (%)", color="#ff7f0e")
    axes[1].set_ylabel("GPU util. (%)")
    axes[1].grid(True, alpha=0.3)

    for key, values in temperatures.items():
        if np.all(np.isnan(values)):
            continue
        axes[2].plot(elapsed, values, label=key.upper())

    axes[2].set_xlabel("Time (minutes)")
    axes[2].set_ylabel("Temperature (°C)")
    axes[2].grid(True, alpha=0.3)
    if temperatures:
        axes[2].legend(loc="upper right")

    axes[0].legend(loc="upper left")
    axes[1].legend(loc="upper left")

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{log_path.stem}.pdf"
    fig.savefig(output_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise Jetson profile logs.")
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "profile",
        help="Directory containing Jetson profile log files (default: profile folder next to this script).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store generated plots (default: same as profile dir).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively in addition to saving them.",
    )
    parser.add_argument(
        "--sensors",
        nargs="*",
        default=None,
        help="Optional list of temperature sensor keys to track (e.g., CPU GPU SOC0 SOC1 SOC2 TJ).",
    )

    args = parser.parse_args()
    profile_dir = args.profile_dir
    output_dir = args.output_dir or profile_dir

    if not profile_dir.exists() or not profile_dir.is_dir():
        raise FileNotFoundError(f"Profile directory not found: {profile_dir}")

    log_files = sorted(profile_dir.glob("*.txt"))
    if not log_files:
        raise FileNotFoundError(f"No profile logs (*.txt) found in {profile_dir}")

    for log_path in log_files:
        data = parse_profile_log(log_path, sensors=args.sensors)
        if data is None:
            continue
        data = trim_after_cpu_drop(data)
        if data is None:
            continue
        plot_profile(log_path, data, output_dir, show=args.show)


if __name__ == "__main__":
    main()

