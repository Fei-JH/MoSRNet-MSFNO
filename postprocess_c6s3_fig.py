import os
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch as MplPatch
from matplotlib.legend_handler import HandlerTuple

from torch.utils.data import DataLoader, TensorDataset

from models.msfno import MSFNO
from models.resnet import ResNet
from models.mosrnet import MoSRNet

import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 28
matplotlib.rcParams['axes.labelsize'] = 24
matplotlib.rcParams['legend.fontsize'] = 24
matplotlib.rcParams['axes.titlesize'] = 28
matplotlib.rcParams['xtick.labelsize'] = 24 
matplotlib.rcParams['ytick.labelsize'] = 24

def C6S3_fig01(
    Intep_cfgpath,                      # interpolation model cfg
    MSFNO_cfgpath,                      # MSFNO cfg (if chosen as predictor)
    ResNet_cfgpath,                     # ResNet cfg (if chosen as predictor)
    predictor="resnet",                 # choose from {"resnet", "msfno", "mosrnet"}
    data_name="beamdi_exp",             # dataset root under ./datasets/
    validdata="beamdi_exp_scut",                   # validation file stem
    baseline="rein",                    # baseline file stem
    dir = r"./postprocessed",           # C4S4-style saving root
    type = "fig",
    loc = "C6S3",
    name = None                         # figure base name; saved as {name}_{validdata}.png
):
    """
    Draw C6S3 figure: stiffness change (%) = (valid - baseline) for both prediction and GT.
    - Keep three cfgs; choose predictor via `predictor`.
    - Save path: {dir}/{loc}/{type}/{name}_{validdata}.png
    """

    # ---------------- helpers ----------------
    model_classes = {"msfno": MSFNO, "resnet": ResNet, "mosrnet": MoSRNet}

    def _build_model(cfg):
        """Instantiate model from cfg['model']."""
        key = cfg["model"]["model"]
        if key not in model_classes:
            raise ValueError(f"Unknown model key in cfg: {key}")
        return model_classes[key](**cfg["model"]["para"])

    def _load_first_pt(cfg):
        """Load the first .pt under {results_path}/model/."""
        mdir = os.path.join(cfg["paths"]["results_path"], "model")
        if not os.path.isdir(mdir):
            raise FileNotFoundError(f"Model dir not found: {mdir}")
        pts = [fn for fn in os.listdir(mdir) if fn.endswith(".pt")]
        if not pts:
            raise FileNotFoundError(f"No .pt found in: {mdir}")
        return os.path.join(mdir, pts[0])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---------------- load cfgs ----------------
    with open(f"./configs/{Intep_cfgpath}", "r") as f:
        Intep_cfg = yaml.load(f, Loader=yaml.SafeLoader)
    with open(f"./configs/{MSFNO_cfgpath}", "r") as f:
        MSFNO_cfg = yaml.load(f, Loader=yaml.SafeLoader)
    with open(f"./configs/{ResNet_cfgpath}", "r") as f:
        ResNet_cfg = yaml.load(f, Loader=yaml.SafeLoader)

    # select predictor cfg by option
    predictor = predictor.lower()
    if predictor not in {"resnet", "msfno", "mosrnet"}:
        raise ValueError("predictor must be one of {'resnet','msfno','mosrnet'}")

    if predictor == "resnet":
        Predictor_cfg = ResNet_cfg
    elif predictor == "msfno":
        Predictor_cfg = MSFNO_cfg

    # ---------------- build & load models ----------------
    # Interpolation model
    intep_model = _build_model(Intep_cfg).to(device)
    intep_pt = _load_first_pt(Intep_cfg)
    intep_state = torch.load(intep_pt, map_location=device, weights_only=True)
    intep_model.load_state_dict(intep_state)
    intep_model.eval()

    # Predictor model
    predictor_model = _build_model(Predictor_cfg).to(device)
    pred_pt = _load_first_pt(Predictor_cfg)
    pred_state = torch.load(pred_pt, map_location=device, weights_only=True)
    predictor_model.load_state_dict(pred_state)
    predictor_model.eval()

    # ---------------- load data ----------------
    valid_path = f"./datasets/{data_name}/{validdata}.pt"
    base_path  = f"./datasets/{data_name}/{baseline}.pt"
    valid_dict = torch.load(valid_path, map_location=device)
    base_dict  = torch.load(base_path,  map_location=device)

    # Expect shapes: mode [N, C(≥4), L], dmg [N, 1, L] (or compatible)
    in_ch = Predictor_cfg["train"]["in_channels"]
    valid_mode = valid_dict["mode"][:, :max(4, in_ch), :]
    base_mode  = base_dict["mode"][:,  :max(4, in_ch), :]

    valid_dmg = valid_dict["dmg"].float()
    base_dmg  = base_dict["dmg"].float()

    # ---------------- inference ----------------
    with torch.no_grad():
        # Interpolation/feature preparation
        out_valid = intep_model(valid_mode.to(device))   # [B, C1, L']
        out_base  = intep_model(base_mode.to(device))    # [B, C1, L']

        # Append normalized 1D grid as an extra channel
        Lp = out_valid.shape[-1]
        grid = torch.linspace(0.0, 1.0, steps=Lp, device=device).reshape(1, 1, -1)
        valid_merged = torch.cat([out_valid, grid.expand(out_valid.size(0), 1, Lp)], dim=1)
        base_merged  = torch.cat([out_base,  grid.expand(out_base.size(0),  1, Lp)], dim=1)

        # Predictor forward
        pred_valid = predictor_model(valid_merged).mean(dim=0, keepdim=True).squeeze().detach().cpu().numpy()
        pred_base  = predictor_model(base_merged ).mean(dim=0, keepdim=True).squeeze().detach().cpu().numpy()

        # to stiffness change in percent (0~100%)
        pred_valid = 100 - pred_valid * 100
        pred_base  = 100 - pred_base  * 100

        # GT of the first sample
        gt_valid = 100 - valid_dmg[0].to(device).squeeze().detach().cpu().numpy() * 100
        gt_base  = 100 - base_dmg[0].to(device).squeeze().detach().cpu().numpy()  * 100

        # relative change (valid - base)
        output = np.array(pred_valid - pred_base)
        gt     = np.array(gt_valid   - gt_base)

    # ---------------- plotting ----------------
    num_points = output.shape[-1]
    x_indices  = np.linspace(0, 5400, num=num_points)
    gt_indices = np.linspace(0, 5400, num=gt.shape[-1])

    pos_mask = output > 0
    neg_mask = ~pos_mask

    fig, ax = plt.subplots(figsize=(10, 8))

    # predicted curve
    ax.plot(x_indices, output, color="#3A3A3A", lw=1, alpha=1.0, zorder=3, label="_nolegend_")
    # fills
    ax.fill_between(x_indices, 0, output, where=pos_mask, facecolor="#EE7E77", alpha=1.0, zorder=2, label="_nolegend_")
    ax.fill_between(x_indices, 0, output, where=neg_mask, facecolor="#68A7BE", alpha=1.0, zorder=2, label="_nolegend_")

    # GT band + step
    ax.fill_between(gt_indices, 0, gt, facecolor="#CFCFCF", alpha=1.0, zorder=1, label="_nolegend_")
    ax.step(gt_indices, gt, color="#3A3A3A", lw=1, linestyle="--", alpha=1.0, where="mid", zorder=4, label="_nolegend_")

    # axes
    ax.axhline(0, color="k", linewidth=1.5, zorder=10)
    ax.set_xlim(0, 5400)
    ax.set_xticks(np.linspace(0, 5400, num=9))
    ax.set_ylim(-60, 160)
    ax.set_yticks(np.linspace(-50, 150, num=9))

    # custom y tick: show absolute values
    def _fmt_y(y, pos):
        return f"{int(abs(y))}"
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_y))

    # region backgrounds and labels
    ax.axhspan(0, 160, color="#feece7", alpha=1.0, zorder=0)   # stiffness loss region
    ax.axhspan(-60, 0, color="#deeeed", alpha=1.0, zorder=0)   # stiffness increase region
    ax.text(150, 142, "Stiffness Loss",     color="#EE7E77",
            va="center", ha="left", fontsize=28, fontweight="bold")
    ax.text(150, -42, "Stiffness Increase", color="#68A7BE",
            va="center", ha="left", fontsize=28, fontweight="bold")

    ax.set_ylabel("Stiffness Change (%)")
    ax.set_xlabel("Beam Span (mm)")

    # legend (tuple for two-color prediction + gray GT)
    red_patch  = MplPatch(color="#EE7E77", edgecolor="none")
    blue_patch = MplPatch(color="#68A7BE", edgecolor="none")
    gt_patch   = MplPatch(facecolor="#CFCFCF", edgecolor="none")

    if predictor == "msfno":
        model_label = "MS-FNO"
    elif predictor == "resnet":
        model_label = "ResNet"
    else:
        raise KeyError (f"no model named {predictor}")
    
    ax.legend(handles=[(red_patch, blue_patch), gt_patch],
              labels=[f"{model_label}'s Prediction", "Theoretical Damage"],
              handler_map={tuple: HandlerTuple(ndivide=None)},
              loc="best")

    ax.grid(True, linestyle="-", alpha=0.3, zorder=0)
    plt.tight_layout()

    # ---------------- save ----------------
    save_path = os.path.join(dir, loc, type)
    os.makedirs(save_path, exist_ok=True)

    # default name if None
    if name is None:
        name = f"{predictor.upper()}-Prediction_vs_GT"

    out_path = os.path.join(save_path, f"{name}_{validdata}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return out_path


# --- User parameters ---
INTEP_CFG = "mosrnet-beamdi_num_t8000-run01-250814-164957.yaml"   # interpolation cfg
MSFNO_CFG = "msfno-beamdi_num_t8000-run01-250814-165002.yaml"    # msfno cfg
RESNET_CFG = "resnet-beamdi_num_t8000-run01-250814-165006.yaml"  # resnet cfg

DATA_NAME = "beamdi_exp"
BASELINE  = "beamdi_exp_rein"
VALID_LIST = ["beamdi_exp_mcut", "beamdi_exp_scut", "beamdi_exp_wcut", "beamdi_exp_wedg"]

SAVE_DIR = "./postprocessed"
TYPE = "fig"
LOC  = "C6S3"

# --- Run for MSFNO predictor ---
for vd in VALID_LIST:
    out_path = C6S3_fig01(
        Intep_cfgpath=INTEP_CFG,
        MSFNO_cfgpath=MSFNO_CFG,
        ResNet_cfgpath=RESNET_CFG,
        predictor="msfno",                    # choose MSFNO as predictor
        data_name=DATA_NAME,
        validdata=vd,
        baseline=BASELINE,
        dir=SAVE_DIR,
        type=TYPE,
        loc=LOC,
        name="Experimental validation results MS-FNO"         # will save as .../MSFNO-Prediction_vs_GT_{vd}.png
    )
    print(f"[MSFNO] Saved: {out_path}")

# --- Run for ResNet predictor ---
for vd in VALID_LIST:
    out_path = C6S3_fig01(
        Intep_cfgpath=INTEP_CFG,
        MSFNO_cfgpath=MSFNO_CFG,
        ResNet_cfgpath=RESNET_CFG,
        predictor="resnet",                    # choose ResNet as predictor
        data_name=DATA_NAME,
        validdata=vd,
        baseline=BASELINE,
        dir=SAVE_DIR,
        type=TYPE,
        loc=LOC,
        name="Experimental validation results ResNet"         # will save as .../RESNET-Prediction_vs_GT_{vd}.png
    )
    print(f"[RESNET] Saved: {out_path}")
