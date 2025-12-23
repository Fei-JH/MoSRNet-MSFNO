"""
Author: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
Date: 2025-08-21 17:33:29
LastEditors: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
LastEditTime: 2025-10-21 15:51:23
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FormatStrFormatter


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


def C5S2_fig01_mode_of_exp(
    data_name="beamdi_exp",
    output_dir="./results/postprocessed",
    output_type="fig",
    output_loc="C5S2",
    output_name="mode_of_exp",
):
    """
    Plot the first three modes for four scenarios with requested customizations:
    - Legend axis has the same width as main subplots.
    - Y-axis tick labels keep one decimal.
    """
    scenarios = ["beamdi_exp_rein", "beamdi_exp_mcut", "beamdi_exp_wcut", "beamdi_exp_wedg"]
    labels = {s: s.split("_")[-1].upper() for s in scenarios}

    styles = {
        "beamdi_exp_rein": {"linestyle": "-", "linewidth": 2, "marker": "o"},
        "beamdi_exp_mcut": {"linestyle": "--", "linewidth": 2, "marker": "s"},
        "beamdi_exp_wcut": {"linestyle": "-.", "linewidth": 2, "marker": "^"},
        "beamdi_exp_wedg": {"linestyle": ":", "linewidth": 2, "marker": "D"},
    }

    device = torch.device("cpu")
    mode_data = {}

    for scenario in scenarios:
        valid_path = f"./datasets/{data_name}/{scenario}.pt"
        if not os.path.isfile(valid_path):
            raise FileNotFoundError(f"Data file not found: {valid_path}")
        valid_dict = torch.load(valid_path, map_location=device)
        mode_t = valid_dict["mode"]
        valid_mode = torch.mean(mode_t, dim=0)[:3, :]
        mode_data[scenario] = valid_mode.detach().cpu().numpy()

    length = next(iter(mode_data.values())).shape[-1]
    x_axis = np.linspace(0.0, 5400.0, num=length)
    xticks = np.linspace(0, 5400, 9)

    fig = plt.figure(figsize=(10, 10), constrained_layout=True)
    grid = fig.add_gridspec(
        nrows=4, ncols=1, height_ratios=[0.25, 1.0, 1.0, 1.0], hspace=0.08
    )
    ax_legend = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[1, 0])
    ax2 = fig.add_subplot(grid[2, 0], sharex=ax1)
    ax3 = fig.add_subplot(grid[3, 0], sharex=ax1)

    def plot_mode(ax, mode_idx):
        handles = []
        for scenario in scenarios:
            y_vals = mode_data[scenario][mode_idx, :]
            line, = ax.plot(x_axis, y_vals, label=labels[scenario], **styles[scenario])
            handles.append(line)
        ax.set_xlim(-100, 5500)
        ax.set_ylabel(None)
        ax.grid(True, alpha=0.3)
        ax.margins(x=0.02, y=0.1)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        return handles

    h1 = plot_mode(ax1, 0)
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_yticks([0.0, 0.5, 1.0])
    ax1.text(0.01, 0.92, "1st Mode", transform=ax1.transAxes, ha="left", va="top")
    ax1.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    _ = plot_mode(ax2, 1)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_yticks([-1.0, 0.0, 1.0])
    ax2.text(0.01, 0.2, "2nd Mode", transform=ax2.transAxes, ha="left", va="top")
    ax2.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    _ = plot_mode(ax3, 2)
    ax3.set_ylim(-1.2, 1.2)
    ax3.set_yticks([-1.0, 0.0, 1.0])
    ax3.text(0.01, 0.2, "3rd Mode", transform=ax3.transAxes, ha="left", va="top")
    ax3.set_xticks(xticks)
    ax3.set_xlabel("Bridge span (mm)")

    ax_legend.axis("off")
    ax_legend.legend(
        handles=h1,
        labels=[labels[s] for s in scenarios],
        loc="center",
        ncol=4,
        frameon=False,
        handlelength=2,
        columnspacing=0.5,
    )

    save_dir = os.path.join(output_dir, output_loc, output_type)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{output_name}.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, save_path


DEFAULT_DATA_NAME = "beamdi_exp"
DEFAULT_OUTPUT_DIR = "./results/postprocessed"
DEFAULT_OUTPUT_TYPE = "fig"
DEFAULT_OUTPUT_LOC = "C5S2"
DEFAULT_OUTPUT_NAME = "mode_of_exp"


def main():
    fig, out_path = C5S2_fig01_mode_of_exp(
        data_name=DEFAULT_DATA_NAME,
        output_dir=DEFAULT_OUTPUT_DIR,
        output_type=DEFAULT_OUTPUT_TYPE,
        output_loc=DEFAULT_OUTPUT_LOC,
        output_name=DEFAULT_OUTPUT_NAME,
    )
    print(f"Figure saved to: {out_path}")


if __name__ == "__main__":
    main()
