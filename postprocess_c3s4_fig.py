"""
Author: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
Date: 2025-08-12 18:06:19
LastEditors: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
LastEditTime: 2025-10-21 15:28:23
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import yaml


def set_plot_style():
    """Apply global Matplotlib style for postprocess figures."""
    matplotlib.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 24,
            "axes.labelsize": 24,
            "legend.fontsize": 24,
            "axes.titlesize": 24,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
        }
    )


set_plot_style()


def C3S4_losscurve(
    model_cfg_path,
    output_dir="./results/postprocessed",
    output_type="fig",
    output_loc="C3S4",
    output_name="Training and validation loss curves of MS-FNO and RseNet",
):
    """Plot training and validation loss curves for a single model config."""
    line_width = 2
    bg_color = "#FFFEFC"

    # --- Load config ---
    with open(f"./configs/{model_cfg_path}", "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    # --- CSV path ---
    model_loss_csv = os.path.join(model_cfg["paths"]["results_path"], "loss/history_output.csv")

    # --- Read CSV ---
    model_df = pd.read_csv(model_loss_csv)

    # --- Check columns ---
    if not {"T_loss", "V_loss"}.issubset(model_df.columns):
        raise KeyError(f"{model_df} CSV missing required columns 'T_loss' and/or 'V_loss'.")

    # --- Extract ---
    model_t = model_df["Tloss_LpLoss.mean"].values
    model_v = model_df["Vloss_LpLoss.mean"].values
    x_indices = range(len(model_t))

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor(bg_color)

    ax.plot(
        x_indices,
        model_t,
        label="Training loss",
        color="k",
        linewidth=line_width,
        zorder=1,
    )
    ax.plot(
        x_indices,
        model_v,
        label="Validation loss",
        color="r",
        linewidth=line_width,
        zorder=2,
    )

    ax.set_xlim(-10, 180)
    ax.set_xticks([0, 20, 45, 70, 95, 120, 145, 170])
    ax.set_ylim(0.001, 2)
    ax.set_yticks([0.001, 0.01, 0.1, 1])
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Relative L2 Norm")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(loc="upper right")

    cfg_lower = model_cfg_path.lower()
    if "mosrnet" in cfg_lower:
        title = "(a)"
    elif "msfno" in cfg_lower:
        title = "(b)"
    elif "resnet" in cfg_lower:
        title = "(c)"
    else:
        title = ""
    ax.set_title("", pad=25)
    ax.set_title(title)

    plt.tight_layout()

    save_path = os.path.join(output_dir, output_loc, output_type)
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(
        os.path.join(save_path, f"{output_name}_{title}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    print(f"Figure saved to {os.path.join(save_path, f'{output_name}_{title}.png')}")
    plt.close(fig)


MSFNO_CFG = "msfno-beamdi_num_t8000-run01-250814-165002.yaml"
RESNET_CFG = "resnet-beamdi_num_t8000-run01-250814-165006.yaml"
MOSRNET_CFG = "mosrnet-beamdi_num_t8000-run01-250814-164957.yaml"


def main():
    for cfg_path in (MOSRNET_CFG, MSFNO_CFG, RESNET_CFG):
        C3S4_losscurve(model_cfg_path=cfg_path)


if __name__ == "__main__":
    main()
