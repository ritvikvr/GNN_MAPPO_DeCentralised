"""Plot algorithm success rate comparison using pandas and matplotlib."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    # Resolve CSV file
    csv_candidates = [
        Path("results.csv"),
        Path("results/episodes_conditions.csv"),
        Path("results/summary_conditions.csv"),
        Path("results/eval.csv")
    ]
    csv_file = None
    for p in csv_candidates:
        if p.exists():
            csv_file = p
            break

    if csv_file is None:
        print("Error: No CSV results file found.")
        sys.exit(1)

    print(f"Reading data from: {csv_file}")
    df = pd.read_csv(csv_file)

    # Resolve algorithm/method column and success rate column
    algo_col = "algorithm" if "algorithm" in df.columns else "method"
    rate_candidates = [
        "success_rate",
        "weighted_completion_rate",
        "total_completion_rate",
        "weighted_completion_rate_mean"
    ]
    rate_col = next((c for c in rate_candidates if c in df.columns), None)

    if algo_col not in df.columns or rate_col is None:
        print(f"Error: Required columns not found in {csv_file}. Columns: {df.columns.tolist()}")
        sys.exit(1)

    print(f"Grouping by '{algo_col}' and computing mean of '{rate_col}'...")
    grouped = df.groupby(algo_col)[rate_col].mean()
    print(grouped)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    grouped.plot(kind="bar", ax=ax, color="#10b981", edgecolor="black", width=0.55)
    ax.set_ylabel("Success Rate")
    ax.set_xlabel("Algorithm")
    ax.set_title("Algorithm Success Rate Comparison")
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=15)
    plt.tight_layout()

    out_file = Path("results/success_rate_comparison.png")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file)
    plt.close()
    print(f"Plot saved successfully to: {out_file}")


if __name__ == "__main__":
    main()
