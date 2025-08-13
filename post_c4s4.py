import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, precision_recall_fscore_support
import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
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
    dir = r"./postprocessed",
    type = "fig",
    loc = "C4S4",
    name = "Training and validation loss curves of MS-FNO and RseNet"
):
        
    msfno_color = "#1f77b4"
    resnet_color = "#d62728"
    linewidth = 1.5

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
    MSFNO_T = MSFNO_df["T_loss"].values
    MSFNO_V = MSFNO_df["V_loss"].values
    ResNet_T = ResNet_df["T_loss"].values
    ResNet_V = ResNet_df["V_loss"].values
    print(MSFNO_T)
    x_msfno_tr = range(len(MSFNO_T))
    x_msfno_va = range(len(MSFNO_V))
    x_resnet_tr = range(len(ResNet_T))
    x_resnet_va = range(len(ResNet_V))

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(24, 12))
    ax_tr, ax_va = axes

    # Training Loss
    ax_tr.plot(x_msfno_tr, MSFNO_T, label="MS-FNO (Training)", color=msfno_color, linewidth=linewidth)
    ax_tr.plot(x_resnet_tr, ResNet_T, label="ResNet (Training)", color=resnet_color, linewidth=linewidth)
    ax_tr.set_xlim(0, 160)
    ax_tr.set_xticks(list(range(0, 151, 25)))
    # ax_tr.set_yscale("log")
    # ax_tr.set_ylim(1e-4, 0.6)
    ax_tr.set_title("Training Loss Curve")
    ax_tr.set_xlabel("Epoch")
    ax_tr.set_ylabel("Loss")
    ax_tr.grid(True, which="both", linestyle="--", alpha=0.3)
    ax_tr.legend()

    # Validation Loss
    ax_va.plot(x_msfno_va, MSFNO_V, label="MS-FNO (Validation)", color=msfno_color, linewidth=linewidth)
    ax_va.plot(x_resnet_va, ResNet_V, label="ResNet (Validation)", color=resnet_color, linewidth=linewidth)
    ax_va.set_xlim(0, 160)
    ax_va.set_xticks(list(range(0, 151, 25)))
    # ax_va.set_yscale("log")
    # ax_va.set_ylim(1e-4, 0.6)
    ax_va.set_title("Validation Loss Curve")
    ax_va.set_xlabel("Epoch")
    ax_va.set_ylabel("Loss")
    ax_va.grid(True, which="both", linestyle="--", alpha=0.3)
    ax_va.legend()

    plt.tight_layout()
    plt.show()
    # os.makedirs(dir, exist_ok=True)
    # out_name = f"{loc}_{type}_MSFNO_ResNet_Loss.png"
    # fig.savefig(os.path.join(dir, out_name), dpi=300, bbox_inches="tight")
    # plt.close(fig)


# %%
MSFNO_cfgpath = "EXP1-MSFNO-BeamDI02_T8000-DD-250721-164537.yaml"
ResNet_cfgpath = "EXP1-ResNet-BeamDI02_T8000-DD-250721-170607.yaml"
MoSRNet_cfgpath = "EXP4-MBCNNSR-BeamDI02_T8000-RD-250724-185445.yaml"

if __name__ == "__main__":
    C4S4_fig01(
        MSFNO_cfgpath=MSFNO_cfgpath,
        ResNet_cfgpath=ResNet_cfgpath,
        MoSRNet_cfgpath=MoSRNet_cfgpath
    )
