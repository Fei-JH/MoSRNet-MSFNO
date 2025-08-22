'''
Author: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
Date: 2025-08-12 18:06:19
LastEditors: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
LastEditTime: 2025-08-22 18:26:54
'''


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, precision_recall_fscore_support
import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 24
matplotlib.rcParams['axes.labelsize'] = 24
matplotlib.rcParams['legend.fontsize'] = 24
matplotlib.rcParams['axes.titlesize'] = 24
matplotlib.rcParams['xtick.labelsize'] = 24 
matplotlib.rcParams['ytick.labelsize'] = 24

#%% Training and validation loss curves of MS-FNO and RseNet
import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt

def C4S4_losscurve(
    model_cfgpath,
    dir = r"./results/postprocessed",
    type = "fig",
    loc = "C4S4",
    name = "Training and validation loss curves of MS-FNO and RseNet"
):
    msfno_color = "#E23C5D"
    resnet_color = "#FFB42C"
    linewidth = 2
    bg_color = "#FFFEFC"  # 浅蓝灰

    # ---- Load configs ----
    with open(f"./configs/{model_cfgpath}", "r") as f:
        model_cfg = yaml.load(f, Loader=yaml.SafeLoader)


    # ---- CSV paths ----
    model_losscsv = os.path.join(model_cfg["paths"]["results_path"], "loss/history_output.csv")
    
    # ---- Read CSV ----
    model_df = pd.read_csv(model_losscsv)

    # ---- Check columns ----
    if not {"T_loss", "V_loss"}.issubset(model_df.columns):
        raise KeyError(f"{model_df} CSV missing required columns 'T_loss' and/or 'V_loss'.")

    # ---- Extract ----
    model_t = model_df["Tloss_LpLoss.mean"].values
    model_v = model_df["Vloss_LpLoss.mean"].values
    
    x_indicies = range(len(model_t))
    

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor(bg_color)

    # plot training loss (black, bottom)
    ax.plot(
        x_indicies, model_t,
        label="Training loss", color="k", linewidth=linewidth, zorder=1
    )

    # plot validation loss (red, upper)
    ax.plot(
        x_indicies, model_v,
        label="Validation loss", color="r", linewidth=linewidth, zorder=2
    )

    # axis formatting
    ax.set_xlim(-10, 180)
    ax.set_xticks([0, 20, 45, 70, 95, 120, 145, 170])
    ax.set_ylim(0.001, 2)
    ax.set_yticks([0.001, 0.01, 0.1, 1])
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Relative L2 Norm")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(loc="upper right")

    # auto title based on config filename
    cfg_lower = model_cfgpath.lower()
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

    save_path = os.path.join(dir, loc, type)
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, f"{name}_{title}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# %%
MSFNO_cfgpath = "msfno-beamdi_num_t8000-run01-250814-165002.yaml"
ResNet_cfgpath = "resnet-beamdi_num_t8000-run01-250814-165006.yaml"
MoSRNet_cfgpath = "mosrnet-beamdi_num_t8000-run01-250814-164957.yaml"

if __name__ == "__main__":
    C4S4_losscurve(
    model_cfgpath=MoSRNet_cfgpath
    )

    C4S4_losscurve(
    model_cfgpath=MSFNO_cfgpath
    )

    C4S4_losscurve(
    model_cfgpath=ResNet_cfgpath
    )
    
