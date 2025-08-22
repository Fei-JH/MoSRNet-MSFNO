'''
Author: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
Date: 2025-08-12 18:06:19
LastEditors: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
LastEditTime: 2025-08-22 15:57:22
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

def C4S4_fig01(
    MSFNO_cfgpath,
    ResNet_cfgpath,
    MoSRNet_cfgpath,
    dir = r"./results/postprocessed",
    type = "fig",
    loc = "C4S4",
    name = "Training and validation loss curves of MS-FNO and RseNet"
):
    msfno_color = "#E23C5D"
    resnet_color = "#FFB42C"
    linewidth = 2.5
    bg_color = "#FFFEFC"  # 浅蓝灰

    # ---- Load configs ----
    with open(f"./configs/{MSFNO_cfgpath}", "r") as f:
        MSFNO_cfg = yaml.load(f, Loader=yaml.SafeLoader)
    with open(f"./configs/{ResNet_cfgpath}", "r") as f:
        ResNet_cfg = yaml.load(f, Loader=yaml.SafeLoader)
    with open(f"./configs/{MoSRNet_cfgpath}", "r") as f:
        MoSRNet_cfg = yaml.load(f, Loader=yaml.SafeLoader)  # not used in plot

    # ---- CSV paths ----
    MSFNO_losscsv = os.path.join(MSFNO_cfg["paths"]["results_path"], "loss/history_output.csv")
    ResNet_losscsv = os.path.join(ResNet_cfg["paths"]["results_path"], "loss/history_output.csv")
    
    # ---- Read CSV ----
    MSFNO_df = pd.read_csv(MSFNO_losscsv)
    ResNet_df = pd.read_csv(ResNet_losscsv)

    # ---- Check columns ----
    for df_name, df in [("MSFNO", MSFNO_df), ("ResNet", ResNet_df)]:
        if not {"T_loss", "V_loss"}.issubset(df.columns):
            raise KeyError(f"{df_name} CSV missing required columns 'T_loss' and/or 'V_loss'.")

    # ---- Extract ----
    MSFNO_T = MSFNO_df["Tloss_LpLoss.mean"].values
    MSFNO_V = MSFNO_df["Vloss_LpLoss.mean"].values
    ResNet_T = ResNet_df["Tloss_LpLoss.mean"].values
    ResNet_V = ResNet_df["Vloss_LpLoss.mean"].values

    x_msfno_tr = range(len(MSFNO_T))
    x_msfno_va = range(len(MSFNO_V))
    x_resnet_tr = range(len(ResNet_T))
    x_resnet_va = range(len(ResNet_V))

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    ax_tr, ax_va = axes

    # 设置背景色
    ax_tr.set_facecolor(bg_color)
    ax_va.set_facecolor(bg_color)

    # Training Loss
    ax_tr.plot(x_msfno_tr, MSFNO_T, label="MS-FNO (Training)", color=msfno_color, linewidth=linewidth)
    ax_tr.plot(x_resnet_tr, ResNet_T, label="ResNet (Training)", color=resnet_color, linewidth=linewidth)
    ax_tr.set_xlim(-10, 180)
    ax_tr.set_xticks([0, 20, 50, 80, 110, 140, 170])
    ax_tr.set_ylim(0.001, 1)
    ax_tr.set_yticks([0.001, 0.01, 0.1, 1])
    ax_tr.set_yscale("log")
    ax_tr.set_title("Training Loss Curve")
    ax_tr.set_xlabel("Epoch")
    ax_tr.set_ylabel(r"Relative L2 Norm")
    ax_tr.grid(True, which="both", linestyle="--", alpha=0.3)
    ax_tr.legend()
    # ax_tr.set_box_aspect(1)

    # Validation Loss
    ax_va.plot(x_msfno_va, MSFNO_V, label="MS-FNO (Validation)", color=msfno_color, linewidth=linewidth)
    ax_va.plot(x_resnet_va, ResNet_V, label="ResNet (Validation)", color=resnet_color, linewidth=linewidth)
    ax_va.set_xlim(-10, 180)
    ax_va.set_xticks([0, 20, 50, 80, 110, 140, 170])
    ax_va.set_ylim(0.001, 1)
    ax_va.set_yticks([0.001, 0.01, 0.1, 1])
    ax_va.set_yscale("log")
    ax_va.set_title("Validation Loss Curve")
    ax_va.set_xlabel("Epoch")
    ax_va.grid(True, which="both", linestyle="--", alpha=0.3)
    ax_va.legend()
    # ax_va.set_box_aspect(1)

    # 去掉 y 轴标题和刻度
    ax_va.set_ylabel("")
    ax_va.set_yticklabels([])

    plt.tight_layout()

    save_path = os.path.join(dir, loc, type)
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, f"{name}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


#%%Training and validation loss curves of MoSRNet

def C4S4_fig02(
    MSFNO_cfgpath,
    ResNet_cfgpath,
    MoSRNet_cfgpath,
    dir = r"./results/postprocessed",
    type = "fig",
    loc = "C4S4",
    name = "Training and validation loss curves of MoSRNet"
):
    mosrnet_color = "#4C72B0"  # 你可以换成喜欢的颜色
    linewidth = 2.5
    bg_color = "#FFFEFC"  # 浅蓝灰

    # ---- Load config ----
    with open(f"./configs/{MoSRNet_cfgpath}", "r") as f:
        MoSRNet_cfg = yaml.load(f, Loader=yaml.SafeLoader)

    # ---- CSV path ----
    MoSRNet_losscsv = os.path.join(MoSRNet_cfg["paths"]["results_path"], "loss/history_output.csv")
    
    # ---- Read CSV ----
    MoSRNet_df = pd.read_csv(MoSRNet_losscsv)

    # ---- Check columns ----
    if not {"T_loss", "V_loss"}.issubset(MoSRNet_df.columns):
        raise KeyError("MoSRNet CSV missing required columns 'T_loss' and/or 'V_loss'.")

    # ---- Extract ----
    MoSRNet_T = MoSRNet_df["T_loss"].values
    MoSRNet_V = MoSRNet_df["V_loss"].values

    x_mosrnet_tr = range(len(MoSRNet_T))
    x_mosrnet_va = range(len(MoSRNet_V))

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    ax_tr, ax_va = axes

    # 设置背景色
    ax_tr.set_facecolor(bg_color)
    ax_va.set_facecolor(bg_color)

    # Training Loss
    ax_tr.plot(x_mosrnet_tr, MoSRNet_T, label="MoSRNet (Training)", color=mosrnet_color, linewidth=linewidth)
    ax_tr.set_xlim(-10, 130)
    ax_tr.set_xticks([0, 20, 40, 60, 80, 100, 120])
    ax_tr.set_ylim(0.001, 2)
    ax_tr.set_yticks([0.001, 0.01, 0.1, 1])
    ax_tr.set_yscale("log")
    ax_tr.set_title("Training Loss Curve")
    ax_tr.set_xlabel("Epoch")
    ax_tr.set_ylabel(r"Relative L2 Norm")  
    ax_tr.grid(True, which="both", linestyle="--", alpha=0.3)
    ax_tr.legend()
    # ax_tr.set_box_aspect(1)

    # Validation Loss
    ax_va.plot(x_mosrnet_va, MoSRNet_V, label="MoSRNet (Validation)", color=mosrnet_color, linewidth=linewidth)
    ax_va.set_xlim(-10, 130)
    ax_va.set_xticks([0, 20, 40, 60, 80, 100, 120])
    ax_va.set_ylim(0.001, 2)
    ax_va.set_yticks([0.001, 0.01, 0.1, 1])
    ax_va.set_yscale("log")
    ax_va.set_title("Validation Loss Curve")
    ax_va.set_xlabel("Epoch")
    ax_va.grid(True, which="both", linestyle="--", alpha=0.3)
    ax_va.legend()
    # ax_va.set_box_aspect(1)

    # 去掉 y 轴标题和刻度（右侧）
    ax_va.set_ylabel("")
    ax_va.set_yticklabels([])

    plt.tight_layout()
    
    save_path = os.path.join(dir, loc, type)
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, f"{name}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# %%
MSFNO_cfgpath = "msfno-beamdi_num_t8000-run01-250814-165002.yaml"
ResNet_cfgpath = "resnet-beamdi_num_t8000-run01-250814-165006.yaml"
MoSRNet_cfgpath = "mosrnet-beamdi_num_t8000-run01-250814-164957.yaml"

if __name__ == "__main__":
    C4S4_fig01(
        MSFNO_cfgpath=MSFNO_cfgpath,
        ResNet_cfgpath=ResNet_cfgpath,
        MoSRNet_cfgpath=MoSRNet_cfgpath
    )

    C4S4_fig02(
        MSFNO_cfgpath=MSFNO_cfgpath,
        ResNet_cfgpath=ResNet_cfgpath,
        MoSRNet_cfgpath=MoSRNet_cfgpath
    )
