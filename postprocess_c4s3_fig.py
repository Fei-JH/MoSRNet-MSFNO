'''
Author: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
Date: 2025-08-12 18:06:19
LastEditors: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
LastEditTime: 2025-10-21 15:48:16
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import os
import pandas as pd
from utilities.mcs_util import generate_sequence

# ---------- Matplotlib global styles ----------
matplotlib.rcParams['font.family'] = 'Helvetica'
matplotlib.rcParams['font.size'] = 28
matplotlib.rcParams['axes.labelsize'] = 24
matplotlib.rcParams['legend.fontsize'] = 24
matplotlib.rcParams['axes.titlesize'] = 28
matplotlib.rcParams['xtick.labelsize'] = 24 
matplotlib.rcParams['ytick.labelsize'] = 24


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
)



def C4S3_dmg_example(
    random_seed=114514,
    dir=r"./results/postprocessed",
    type="csv",
    loc="C4S3",
    name="C4S3-MS-FNO_SignedError+Intact",
    n_elem=540,
    y=10  # same meaning as in generate_sequence
):
    # set global font
    plt.rcParams['font.family'] = 'Times New Roman'

    # set random seed
    np.random.seed(random_seed)

    # --- generate random damage sequence ---
    dmg, dmgloc, dgrs = generate_sequence(
        length=n_elem,
        y=y,
        noise_range=0.05,
        dip_range=(0.15, 0.6)
    )

    # --- create save directory for figures ---
    fig_dir = os.path.join(dir, loc, "fig")
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, f"{name}_dmg_distribution.png")

    # --- plot damage distribution ---
    x_indices = np.linspace(0, 5400, n_elem)
    fig, ax = plt.subplots(figsize=(10, 7))

    # plot the damage line
    ax.plot(x_indices, 100 - 100 * dmg, color="#000000", linewidth=1.5,
            label="Damage field", zorder=2)

    # add Y=0 reference line
    ax.axhline(0, color="black", linestyle="-", linewidth=2, zorder=1)

    min_idx = np.argmin(dmg)  # find the index of minimum damage value
    x_pos = x_indices[min_idx]
    ax.axvline(x_pos+5, color="red", linestyle="--", linewidth=1.5,
            label="Damage location", zorder=3)


    # axis settings
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


def C4S3_plot_cm(
    y_true, y_pred, n_bins, save_path=None, font_size=14, title="", show=True
):
    """
    Plot confusion matrix with fixed bin space and compute metrics.

    Key points:
    - Keep all bins in the heatmap (labels fixed as 0..n_bins-1).
    - Row-wise percentage normalization is robust to zero-division.
    - Macro metrics are averaged ONLY over classes with support > 0 (valid-class macro).

    Args:
        y_true (np.ndarray): Ground-truth class indices (0..n_bins-1).
        y_pred (np.ndarray): Predicted class indices (0..n_bins-1).
        n_bins (int): Number of bins (fixed label space for plotting).
        save_path (str or None): If provided, save figure to this path.
        font_size (int): Font size for annotations and ticks.
        title (str): Optional figure title.
        show (bool): Whether to display the figure (plt.show()).

    Returns:
        (acc, macro_prec, macro_recall, macro_f1): tuple of floats in [0,1].
            - macro_* are averaged over classes with support > 0 ONLY.
    """
    # ---- Fixed label space to keep all bins in the plot ----
    labels = np.arange(n_bins)

    # ---- Human-readable tick labels (start from 1), plus "R"/"P" row/col ----
    class_labels = [str(i + 1) for i in range(n_bins)]
    xticklabels = class_labels + ['R']  # last column shows precision
    yticklabels = class_labels + ['P']  # last row shows recall

    # ---- Confusion matrix with fixed labels (keeps empty classes as zero rows) ----
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # ---- Robust row-wise normalization to percentages (no warnings on zero rows) ----
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_percent = np.divide(
        cm.astype(float) * 100.0,
        row_sums,
        out=np.zeros_like(cm, dtype=float),  # fill 0% for empty rows
        where=row_sums != 0
    )

    # ---- Per-class metrics; VALID-CLASS MACRO over classes with support > 0 only ----
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    mask = supports > 0  # only classes appearing in y_true are considered "valid"
    if mask.any():
        macro_prec = float(np.mean(precisions[mask]))
        macro_recall = float(np.mean(recalls[mask]))
        macro_f1 = float(np.mean(f1s[mask]))
    else:
        macro_prec = macro_recall = macro_f1 = 0.0

    # ---- Overall accuracy ----
    acc = accuracy_score(y_true, y_pred)

    # ---- Build cell annotations for main n_bins x n_bins block ----
    annot = np.empty(cm.shape, dtype=object)
    for i in range(n_bins):
        for j in range(n_bins):
            count = cm[i, j]
            percent = cm_percent[i, j]
            # Show "count\nxx.x%" in each cell
            annot[i, j] = f"{count}\n{percent:.1f}%"

    # ---- Extend to (n_bins+1) x (n_bins+1): last row/col for P/R and bottom-right for Acc ----
    cm_full = np.zeros((n_bins + 1, n_bins + 1), dtype=float)
    cm_full[:-1, :-1] = cm_percent
    annot_full = np.empty((n_bins + 1, n_bins + 1), dtype=object)
    annot_full[:-1, :-1] = annot

    # Column-wise precision info at the bottom row
    for j in range(n_bins):
        col_sum = cm[:, j].sum()
        precision_percent = precisions[j] * 100.0
        annot_full[-1, j] = f"{col_sum}\n{precision_percent:.1f}%"
        cm_full[-1, j] = precision_percent  # color scale

    # Row-wise recall info at the rightmost column
    for i in range(n_bins):
        row_sum = cm[i, :].sum()
        recall_percent = recalls[i] * 100.0
        annot_full[i, -1] = f"{row_sum}\n{recall_percent:.1f}%"
        cm_full[i, -1] = recall_percent

    # Bottom-right shows overall accuracy (as percentage)
    annot_full[-1, -1] = f"Acc:\n{acc * 100.0:.1f}%"
    cm_full[-1, -1] = acc * 100.0

    # ---- Plotting ----
    fig, ax = plt.subplots(figsize=(12, 10))

    # Disable per-cell text if too many bins; otherwise annotate with bold font
    if n_bins > 12:
        annot_heatmap = False
        annot_kws = None
    else:
        annot_heatmap = annot_full
        annot_kws = {"size": font_size, "va": "center", "fontweight": "bold", "font": "Helvetica"}

    sns.heatmap(
        cm_full,
        annot=annot_heatmap,
        fmt='',
        cmap="GnBu",
        cbar=False,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        linewidths=0.25,
        linecolor='black',
        square=True,
        annot_kws=annot_kws if annot_kws is not None else {}
    )

    # Green rectangle on the main diagonal cells
    for i in range(n_bins):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='green', lw=2))

    # Axis labels and ticks
    axis_fontsize = font_size + 3
    label_fontsize = font_size + 3
    ax.set_xlabel(
        "Predicted Damage Location (Element Index)",
        fontsize=axis_fontsize + 2, weight='bold', labelpad=5
    )
    ax.set_ylabel(
        "Actual Damage Location (Element Index)",
        fontsize=axis_fontsize + 2, weight='bold', labelpad=5
    )
    ax.set_xticklabels(xticklabels, fontsize=label_fontsize, rotation=0, ha='center', weight='normal')
    ax.set_yticklabels(yticklabels, fontsize=label_fontsize, rotation=0, va='center', weight='normal')


    # Colorbar aligned with the heatmap
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.3)
    im = ax.collections[0]
    cb = plt.colorbar(im, cax=cax, orientation='vertical')
    cb.ax.tick_params(labelsize=font_size + 3, width=1.7, length=6, labelcolor='black')
    cb.ax.set_ylim(0, 100)
    im.set_cmap('GnBu')
    im.set_clim(0, 100)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close()

    return acc, macro_prec, macro_recall, macro_f1


if __name__ == "__main__":
    C4S3_dmg_example(
        random_seed=114514,
        dir=r"./results/postprocessed",
        type="csv",
        loc="C4S3",
        name="C4S3-dmg_example",
        n_elem=540)


    n_elem = 540
    left_skip = 108
    right_skip = 108
    valid_start = left_skip
    valid_end = n_elem - right_skip

    for n in [10, 15, 20]:
        n_bins = n  
        bin_edges = np.linspace(0, n_elem, n_bins + 1, dtype=int)

        def digitize_pos(pos_list):
            pos_bin = np.digitize(pos_list, bin_edges) - 1
            pos_bin = np.clip(pos_bin, 0, n_bins - 1)
            return pos_bin

        df = pd.read_csv("./results/mcs_test/msc_maxpos_rawidx.csv")
        df80 = df[(df["true_idx"] >= valid_start) & (df["true_idx"] < valid_end)].reset_index(drop=True)

        # ==========all elements==========
        true_bin = digitize_pos(df["true_idx"].values)
        msfno_bin = digitize_pos(df["msfno_idx"].values)
        resnet_bin = digitize_pos(df["resnet_idx"].values)
        msfno_int_bin = digitize_pos(df["msfno_int_idx"].values)
        resnet_int_bin = digitize_pos(df["resnet_int_idx"].values)

        cases = {
            "msfno": msfno_bin,
            "resnet": resnet_bin,
            "msfno_int": msfno_int_bin,
            "resnet_int": resnet_int_bin
        }
        true_labels = true_bin

        metrics_dict = {
            "Acc": {},
            "Prec": {},
            "Rec": {},
            "F1": {}
        }

        output_dir = "./results/postprocessed"
        os.makedirs(output_dir, exist_ok=True)

        for case_name, pred_bin in cases.items():
            fig_name = f"C4S3_confmat_{case_name}_bin={n_bins}.png"
            fig_path = os.path.join(output_dir, "C4S3", "fig")
            os.makedirs(fig_path, exist_ok=True)
            acc, prec, rec, f1 = C4S3_plot_cm(
                true_labels, pred_bin, n, 
                save_path=os.path.join(fig_path, fig_name), 
                font_size=14,
                title=f"{case_name.upper()}, bin={n_bins}", show=False
            )
            metrics_dict["Acc"][case_name] = acc
            metrics_dict["Prec"][case_name] = prec
            metrics_dict["Rec"][case_name] = rec
            metrics_dict["F1"][case_name] = f1

        metrics_df = pd.DataFrame(metrics_dict)
        metrics_df = metrics_df.T  # 行为指标
        csv_path = os.path.join(output_dir, "C4S3", "csv")
        os.makedirs(csv_path, exist_ok=True)
        metrics_df.to_csv(os.path.join(csv_path, f"C4S3_main_metrics_bin={n_bins}.csv") )

        print(f"Confusion matrixes and csv file saved, bin={n_bins}")


        # ==========central 80% elements==========
        true_bin = digitize_pos(df80["true_idx"].values)
        msfno_bin = digitize_pos(df80["msfno_idx"].values)
        resnet_bin = digitize_pos(df80["resnet_idx"].values)
        msfno_int_bin = digitize_pos(df80["msfno_int_idx"].values)
        resnet_int_bin = digitize_pos(df80["resnet_int_idx"].values)

        cases = {
            "msfno": msfno_bin,
            "resnet": resnet_bin,
            "msfno_int": msfno_int_bin,
            "resnet_int": resnet_int_bin
        }
        true_labels = true_bin

        metrics_dict = {
            "Acc": {},
            "Prec": {},
            "Rec": {},
            "F1": {}
        }

        output_dir = "./results/postprocessed"
        os.makedirs(output_dir, exist_ok=True)

        for case_name, pred_bin in cases.items():
            fig_name = f"C4S3_confmat_{case_name}_c60_bin={n_bins}.png"
            fig_path = os.path.join(output_dir, "C4S3", "fig")
            os.makedirs(fig_path, exist_ok=True)
            acc, prec, rec, f1 = C4S3_plot_cm(
                true_labels, pred_bin, n,
                save_path=os.path.join(fig_path, fig_name), 
                font_size=14,
                title=f"{case_name.upper()}, bin={n_bins}", show=False
            )
            metrics_dict["Acc"][case_name] = acc
            metrics_dict["Prec"][case_name] = prec
            metrics_dict["Rec"][case_name] = rec
            metrics_dict["F1"][case_name] = f1

        metrics_df80 = pd.DataFrame(metrics_dict)
        metrics_df80 = metrics_df80.T  # 行为指标
        csv_path = os.path.join(output_dir, "C4S3", "csv")
        os.makedirs(csv_path, exist_ok=True)
        metrics_df80.to_csv(os.path.join(csv_path, f"C4S3_main_metrics_c60_bin={n_bins}.csv") )

        print(f"Confusion matrixes and csv file saved, bin={n_bins}")
    # %%
