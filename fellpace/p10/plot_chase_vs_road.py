"""Plot Hallam Chase times against 5km and 10km best times from the combined p10 dataset."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


INPUT_CSV = Path("DB/po10_chase_combined.csv")
OUTPUT_PNG = Path("po10_chase_vs_road.png")


def plot_chase_vs_road_times(
    input_csv: Path = INPUT_CSV,
    output_png: Path = OUTPUT_PNG,
) -> Path:
    """Create scatter plot with Chase time (Y) vs road times (X) for 5km and 10km series."""
    df = pd.read_csv(input_csv)

    required = {"chase_time_seconds", "5km_best_seconds", "10km_best_seconds"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {input_csv}: {sorted(missing)}")

    fig, ax = plt.subplots(figsize=(10, 7))

    five_df = df.dropna(subset=["5km_best_seconds", "chase_time_seconds"])
    ten_df = df.dropna(subset=["10km_best_seconds", "chase_time_seconds"])

    ax.scatter(
        five_df["5km_best_seconds"],
        five_df["chase_time_seconds"],
        alpha=0.75,
        s=40,
        label=f"5km ({len(five_df)} points)",
        color="#1f77b4",
    )

    ax.scatter(
        ten_df["10km_best_seconds"],
        ten_df["chase_time_seconds"],
        alpha=0.75,
        s=40,
        label=f"10km ({len(ten_df)} points)",
        color="#ff7f0e",
    )

    ax.set_xlabel("Road best time (seconds)")
    ax.set_ylabel("Hallam Chase time (seconds)")
    ax.set_title("Hallam Chase vs Road Best Times (all years combined)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)

    return output_png


def main() -> None:
    output_path = plot_chase_vs_road_times()
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
