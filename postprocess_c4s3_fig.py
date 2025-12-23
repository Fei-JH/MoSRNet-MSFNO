"""
Author: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
Date: 2025-08-12 18:06:19
LastEditors: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
LastEditTime: 2025-10-21 15:48:16
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from utilities.mcs_util import generate_sequence


def set_plot_style():
    """Apply global Matplotlib style for postprocess figures."""
    matplotlib.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 28,
            "axes.labelsize": 24,
            "legend.fontsize": 24,
            "axes.titlesize": 28,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
        }
    )


set_plot_style()


def C4S3_dmg_example(
    random_seed=114514,
    output_dir="./results/postprocessed",
    output_type="csv",
    output_loc="C4S3",
    output_name="C4S3-dmg_example",
    n_elem=540,
    y=10,
):
    """Plot a sample damage distribution and save the figure."""
    np.random.seed(random_seed)

    dmg, _, _ = generate_sequence(length=n_elem, y=y, noise_range=0.05, dip_range=(0.15, 0.6))

    fig_dir = os.path.join(output_dir, output_loc, "fig")
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, f"{output_name}_dmg_distribution.png")

    x_indices = np.linspace(0, 5400, n_elem)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(
        x_indices,
        100 - 100 * dmg,
        color="#000000",
        linewidth=1.5,
        label="Damage field",
        zorder=2,
    )
    ax.axhline(0, color="black", linestyle="-", linewidth=2, zorder=1)

    min_idx = np.argmin(dmg)
    x_pos = x_indices[min_idx]
    ax.axvline(x_pos + 5, color="red", linestyle="--", linewidth=1.5, label="Damage location", zorder=3)

    ax.set_xlim(0, 5400)
    ax.set_xticks(np.linspace(0, 5400, num=9))
    ax.set_ylim(-20, 65)
    ax.set_yticks(np.linspace(-15, 60, num=6))
    ax.set_ylabel("Stiffness Loss (%)")
    ax.set_xlabel("Beam Span (mm)")

    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)

    print(f"Damage plot saved to: {fig_path}")


def C4S3_plot_cm(y_true, y_pred, n_bins, save_path=None, font_size=14, title="", show=True):
    """
    Plot confusion matrix with fixed bin space and compute metrics.
    """
    labels = np.arange(n_bins)
    class_labels = [str(i + 1) for i in range(n_bins)]
    xticklabels = class_labels + ["R"]
    yticklabels = class_labels + ["P"]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_percent = np.divide(
        cm.astype(float) * 100.0,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )

    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    mask = supports > 0
    if mask.any():
        macro_prec = float(np.mean(precisions[mask]))
        macro_recall = float(np.mean(recalls[mask]))
        macro_f1 = float(np.mean(f1s[mask]))
    else:
        macro_prec = macro_recall = macro_f1 = 0.0

    acc = accuracy_score(y_true, y_pred)

    annot = np.empty(cm.shape, dtype=object)
    for i in range(n_bins):
        for j in range(n_bins):
            count = cm[i, j]
            percent = cm_percent[i, j]
            annot[i, j] = f"{count}\n{percent:.1f}%"

    cm_full = np.zeros((n_bins + 1, n_bins + 1), dtype=float)
    cm_full[:-1, :-1] = cm_percent
    annot_full = np.empty((n_bins + 1, n_bins + 1), dtype=object)
    annot_full[:-1, :-1] = annot

    for j in range(n_bins):
        col_sum = cm[:, j].sum()
        precision_percent = precisions[j] * 100.0
        annot_full[-1, j] = f"{col_sum}\n{precision_percent:.1f}%"
        cm_full[-1, j] = precision_percent

    for i in range(n_bins):
        row_sum = cm[i, :].sum()
        recall_percent = recalls[i] * 100.0
        annot_full[i, -1] = f"{row_sum}\n{recall_percent:.1f}%"
        cm_full[i, -1] = recall_percent

    annot_full[-1, -1] = f"Acc:\n{acc * 100.0:.1f}%"
    cm_full[-1, -1] = acc * 100.0

    fig, ax = plt.subplots(figsize=(12, 10))
    if n_bins > 12:
        annot_heatmap = False
        annot_kws = None
    else:
        annot_heatmap = annot_full
        annot_kws = {"size": font_size, "va": "center", "fontweight": "bold", "font": "Helvetica"}

    sns.heatmap(
        cm_full,
        annot=annot_heatmap,
        fmt="",
        cmap="GnBu",
        cbar=False,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        linewidths=0.25,
        linecolor="black",
        square=True,
        annot_kws=annot_kws if annot_kws is not None else {},
    )

    for i in range(n_bins):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor="green", lw=2))

    axis_fontsize = font_size + 3
    label_fontsize = font_size + 3
    ax.set_xlabel(
        "Predicted Damage Location (Element Index)",
        fontsize=axis_fontsize + 2,
        weight="bold",
        labelpad=5,
    )
    ax.set_ylabel(
        "Actual Damage Location (Element Index)",
        fontsize=axis_fontsize + 2,
        weight="bold",
        labelpad=5,
    )
    ax.set_xticklabels(xticklabels, fontsize=label_fontsize, rotation=0, ha="center", weight="normal")
    ax.set_yticklabels(yticklabels, fontsize=label_fontsize, rotation=0, va="center", weight="normal")

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.3)
    im = ax.collections[0]
    cb = plt.colorbar(im, cax=cax, orientation="vertical")
    cb.ax.tick_params(labelsize=font_size + 3, width=1.7, length=6, labelcolor="black")
    cb.ax.set_ylim(0, 100)
    im.set_cmap("GnBu")
    im.set_clim(0, 100)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close()

    return acc, macro_prec, macro_recall, macro_f1


def digitize_pos(pos_list, bin_edges, n_bins):
    pos_bin = np.digitize(pos_list, bin_edges) - 1
    return np.clip(pos_bin, 0, n_bins - 1)


DEFAULT_OUTPUT_DIR = "./results/postprocessed"
DEFAULT_OUTPUT_LOC = "C4S3"
DEFAULT_N_ELEM = 540
DEFAULT_LEFT_SKIP = 108
DEFAULT_RIGHT_SKIP = 108
DEFAULT_N_BINS_LIST = [10, 15, 20]


def main():
    C4S3_dmg_example(
        random_seed=114514,
        output_dir=DEFAULT_OUTPUT_DIR,
        output_type="csv",
        output_loc=DEFAULT_OUTPUT_LOC,
        output_name="C4S3-dmg_example",
        n_elem=DEFAULT_N_ELEM,
    )

    valid_start = DEFAULT_LEFT_SKIP
    valid_end = DEFAULT_N_ELEM - DEFAULT_RIGHT_SKIP

    df = pd.read_csv("./results/mcs_test/msc_maxpos_rawidx.csv")
    df80 = df[(df["true_idx"] >= valid_start) & (df["true_idx"] < valid_end)].reset_index(drop=True)

    for n_bins in DEFAULT_N_BINS_LIST:
        bin_edges = np.linspace(0, DEFAULT_N_ELEM, n_bins + 1, dtype=int)

        true_bin = digitize_pos(df["true_idx"].values, bin_edges, n_bins)
        msfno_bin = digitize_pos(df["msfno_idx"].values, bin_edges, n_bins)
        resnet_bin = digitize_pos(df["resnet_idx"].values, bin_edges, n_bins)
        msfno_int_bin = digitize_pos(df["msfno_int_idx"].values, bin_edges, n_bins)
        resnet_int_bin = digitize_pos(df["resnet_int_idx"].values, bin_edges, n_bins)

        cases = {
            "msfno": msfno_bin,
            "resnet": resnet_bin,
            "msfno_int": msfno_int_bin,
            "resnet_int": resnet_int_bin,
        }

        metrics_dict = {"Acc": {}, "Prec": {}, "Rec": {}, "F1": {}}
        output_dir = DEFAULT_OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)

        for case_name, pred_bin in cases.items():
            fig_name = f"C4S3_confmat_{case_name}_bin={n_bins}.png"
            fig_path = os.path.join(output_dir, "C4S3", "fig")
            os.makedirs(fig_path, exist_ok=True)
            acc, prec, rec, f1 = C4S3_plot_cm(
                true_bin,
                pred_bin,
                n_bins,
                save_path=os.path.join(fig_path, fig_name),
                font_size=14,
                title=f"{case_name.upper()}, bin={n_bins}",
                show=False,
            )
            metrics_dict["Acc"][case_name] = acc
            metrics_dict["Prec"][case_name] = prec
            metrics_dict["Rec"][case_name] = rec
            metrics_dict["F1"][case_name] = f1

        metrics_df = pd.DataFrame(metrics_dict)
        metrics_df = metrics_df.T
        csv_path = os.path.join(output_dir, "C4S3", "csv")
        os.makedirs(csv_path, exist_ok=True)
        metrics_df.to_csv(os.path.join(csv_path, f"C4S3_main_metrics_bin={n_bins}.csv"))
        print(f"Confusion matrixes and csv file saved, bin={n_bins}")

        true_bin = digitize_pos(df80["true_idx"].values, bin_edges, n_bins)
        msfno_bin = digitize_pos(df80["msfno_idx"].values, bin_edges, n_bins)
        resnet_bin = digitize_pos(df80["resnet_idx"].values, bin_edges, n_bins)
        msfno_int_bin = digitize_pos(df80["msfno_int_idx"].values, bin_edges, n_bins)
        resnet_int_bin = digitize_pos(df80["resnet_int_idx"].values, bin_edges, n_bins)

        cases = {
            "msfno": msfno_bin,
            "resnet": resnet_bin,
            "msfno_int": msfno_int_bin,
            "resnet_int": resnet_int_bin,
        }

        metrics_dict = {"Acc": {}, "Prec": {}, "Rec": {}, "F1": {}}
        for case_name, pred_bin in cases.items():
            fig_name = f"C4S3_confmat_{case_name}_c60_bin={n_bins}.png"
            fig_path = os.path.join(output_dir, "C4S3", "fig")
            os.makedirs(fig_path, exist_ok=True)
            acc, prec, rec, f1 = C4S3_plot_cm(
                true_bin,
                pred_bin,
                n_bins,
                save_path=os.path.join(fig_path, fig_name),
                font_size=14,
                title=f"{case_name.upper()}, bin={n_bins}",
                show=False,
            )
            metrics_dict["Acc"][case_name] = acc
            metrics_dict["Prec"][case_name] = prec
            metrics_dict["Rec"][case_name] = rec
            metrics_dict["F1"][case_name] = f1

        metrics_df80 = pd.DataFrame(metrics_dict)
        metrics_df80 = metrics_df80.T
        csv_path = os.path.join(output_dir, "C4S3", "csv")
        os.makedirs(csv_path, exist_ok=True)
        metrics_df80.to_csv(os.path.join(csv_path, f"C4S3_main_metrics_c60_bin={n_bins}.csv"))
        print(f"Confusion matrixes and csv file saved, bin={n_bins}")


if __name__ == "__main__":
    main()
