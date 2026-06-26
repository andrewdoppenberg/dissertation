from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  # only if you need it elsewhere

"""
Soul-Making Data Visualization Module

This script holds plotting logic to generate final PDFs comparing different 
virtue distribution models across multiple suffering and horror regimes.
"""

def generate_plots_from_pkl(path):
    """
    Generate all line and delta bar plots for Soul-Making results from the pickle at `path`.
    Outputs PDFs under <csv-parent>/figures/<date-from-filename>/.
    Returns a list of saved file paths (strings).
    """
    pkl_path = Path(path)
    if not pkl_path.exists():
        raise FileNotFoundError(f"No such file: {pkl_path}")

    # Create the figures in ../figures
    output_dir = Path(__file__).resolve().parent.parent / "figures"
    os.makedirs(output_dir, exist_ok=True)

    # Matplotlib settings
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Garamond"]
    plt.rcParams.update({"font.size": 16})

    # Load data
    data = pd.read_pickle(pkl_path)
    df1 = data["T_grouped"]
    df2 = data["best_by_bias"]

    # Columns
    x_col = "event_bias"
    group_cols = ["dist", "hor_rate"]

    equation_names = {
        1: "Moderation",
        2: "Extremization",
        3: "Desperation",
        4: "Growth",
        5: "Habituation",
        6: "Regression"
    }

    saved = []

    # Helper for safe labels
    def _safe_val(val):
        if isinstance(val, float):
            return f"{val:.2f}".replace(".", "p")
        return str(val).replace(" ", "_").replace("/", "_")

    # Line plots grouped by (dist, hor_rate) and split by equation
    for params, group in df1.groupby(group_cols):
        param_label = "_".join(f"{col}-{_safe_val(val)}" for col, val in zip(group_cols, params))

        for col, ylab, suffix in [
            ("Final Happiness", "Final Happiness", "H"),
            ("Final Virtue", "Final Virtue", "V"),
            ("Final Difference", "Final Virtue and Happiness Difference", "D"),
            ("Best Final Happiness", "Final Happiness", "HB"),
            ("Best Final Virtue", "Final Virtue", "VB"),
            ("Best Final Difference", "Final Virtue and Happiness Difference", "DB"),
            ("Horrors", "Number of Horrendous Events", "HOR"),
            ("Horror Losses", "Average Resources Lost during Horrors", "HL"),
            ("Can Help Rate", "Can Help Rate (%)", "CHR"),
            ("Help Rate", "Help Rate (%)", "HR"),
            ("Help Resources", "Average Resources Exchanged when Helping", "RH"),
            ("Help Virtue", "Average Virtue Gained when Helping", "VH"),
        ]:
            plt.figure()
            for eq_num, subgrp in group.groupby("equation"):
                eq_label = equation_names.get(int(eq_num), f"Equation {eq_num}")
                plt.plot(subgrp[x_col], subgrp[col], label=eq_label)
            plt.xlabel("Natural Evil Level")
            plt.ylabel(ylab)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            outpath = output_dir / f"{param_label}-{suffix}.pdf"
            plt.savefig(outpath)
            plt.close()
            saved.append(str(outpath))

    # ---- Delta bar charts ----
    def _make_diff_bar_from_pivot(sub, index_col, col_col, value_col, lhs, rhs, title, ylabel, out_path):
        pivot = sub.pivot_table(index=index_col, columns=col_col, values=value_col, aggfunc="mean").sort_index()
        for k in [lhs, rhs]:
            if k not in pivot.columns:
                pivot[k] = float("nan")
        diff = pivot[lhs] - pivot[rhs]
        if diff.dropna().empty:
            raise ValueError(
                f"No comparable data for {out_path.name}: missing {lhs!r} or {rhs!r} "
                f"in {col_col} for {value_col}."
            )
        x = diff.index.values
        y = diff.values
        x_labels = [equation_names.get(int(eq), f"Equation {eq}") for eq in x]
        plt.figure()
        plt.bar(x, y)
        plt.xticks(x, x_labels, rotation=45)
        plt.xlabel("Equation")
        plt.ylabel(ylabel)
        # Break the parenthetical qualifier onto a second line so longer
        # distribution names fit within the figure width instead of clipping.
        plt.title(title.replace(" (", "\n(", 1))
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        saved.append(str(out_path))

    # Ensure numeric equation col
    df2["equation"] = pd.to_numeric(df2["equation"], errors="coerce")

    # 1) dist=random; (hor_rate=0.01) − (hor_rate=0.00) for Best Mean Final Virtue
    d1 = df2[(df2["dist"] == "random") & (df2["hor_rate"].isin([0.0, 0.01]))]
    _make_diff_bar_from_pivot(
        d1, "equation", "hor_rate", "Best Mean Final Virtue", 0.01, 0.0,
        "Δ Best Mean Final Virtue (Horrors)", "Change in Performance",
        output_dir / "delta_V_1.pdf",
    )

    # 2) dist=random; (hor_rate=0.00) − (hor_rate=0.01) for Best Mean Final Difference
    d2 = df2[(df2["dist"] == "random") & (df2["hor_rate"].isin([0.0, 0.01]))]
    _make_diff_bar_from_pivot(
        d2, "equation", "hor_rate", "Best Mean Final Difference", 0.0, 0.01,
        "Δ Best Mean Final Difference (Horrors)", "Change in Performance",
        output_dir / "delta_D_1.pdf",
    )

    # 3) dist=random; (hor_rate=0.50) − (hor_rate=0.00) for Best Mean Final Virtue
    d3 = df2[(df2["dist"] == "random") & (df2["hor_rate"].isin([0.0, 0.5]))]
    _make_diff_bar_from_pivot(
        d3, "equation", "hor_rate", "Best Mean Final Virtue", 0.5, 0.0,
        "Δ Best Mean Final Virtue (Horrors)", "Change in Performance",
        output_dir / "delta_V_2.pdf",
    )

    # 4) dist=random; (hor_rate=0.00) − (hor_rate=0.50) for Best Mean Final Difference
    d4 = df2[(df2["dist"] == "random") & (df2["hor_rate"].isin([0.0, 0.5]))]
    _make_diff_bar_from_pivot(
        d4, "equation", "hor_rate", "Best Mean Final Difference", 0.0, 0.5,
        "Δ Best Mean Final Difference (Horrors)", "Change in Performance",
        output_dir / "delta_D_2.pdf",
    )

    # 5) hor_rate=0.01; (fair) − (random) for Best Mean Final Virtue
    d5 = df2[(df2["hor_rate"] == 0.01) & (df2["dist"].isin(["random", "fair"]))]

    _make_diff_bar_from_pivot(
        d5, "equation", "dist", "Best Mean Final Virtue", "fair", "random",
        "Δ Best Mean Final Virtue (Happiness-Targetting Distribution)", "Change in Performance",
        output_dir / "delta_V_3.pdf",
    )

    # 6) hor_rate=0.01; (random) − (fair) for Best Mean Final Difference
    d6 = df2[(df2["hor_rate"] == 0.01) & (df2["dist"].isin(["random", "fair"]))]

    _make_diff_bar_from_pivot(
        d6, "equation", "dist", "Best Mean Final Difference", "random", "fair",
        "Δ Best Mean Final Difference (Happiness-Targetting Distribution)", "Change in Performance",
        output_dir / "delta_D_3.pdf",
    )

    # 7) hor_rate=0.01; (lowVirtue) - (random) for Best Mean Final Virtue
    d7 = df2[(df2["hor_rate"] == 0.01) & (df2["dist"].isin(["random", "lowVirtue"]))]

    _make_diff_bar_from_pivot(
        d7, "equation", "dist", "Best Mean Final Virtue", "lowVirtue", "random",
        "Δ Best Mean Final Virtue (Virtue-Targetting Distribution)", "Change in Performance",
        output_dir / "delta_V_4.pdf",
    )

    # 8) hor_rate=0.01; (random) - (lowVirtue) for Best Mean Final Difference
    d8 = df2[(df2["hor_rate"] == 0.01) & (df2["dist"].isin(["random", "lowVirtue"]))]

    _make_diff_bar_from_pivot(
        d8, "equation", "dist", "Best Mean Final Difference", "random", "lowVirtue",
        "Δ Best Mean Final Difference (Virtue-Targetting Distribution)", "Change in Performance",
        output_dir / "delta_D_4.pdf",
    )

    return saved

if __name__ == "__main__":
    # Automatically look for the pickle file in the ../results directory
    pkl_file_path = Path(__file__).resolve().parent.parent / "results" / "soul-making.pkl"
    if pkl_file_path.exists():
        print(f"Found results file at: {pkl_file_path}\nGenerating plots...")
        try:
            saved_files = generate_plots_from_pkl(str(pkl_file_path))
            print(f"Successfully generated {len(saved_files)} plots.")
            print("\n".join(saved_files))
        except Exception as e:
            print(f"Error generating plots: {e}")
    else:
        print(f"Could not find results file at: {pkl_file_path}")
