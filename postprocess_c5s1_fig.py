import os
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch as MplPatch
from matplotlib.legend_handler import HandlerTuple

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

# ---------------------------- helpers ----------------------------

MODEL_CLASSES = {"msfno": MSFNO, "resnet": ResNet, "mosrnet": MoSRNet}

def _build_model_from_cfg(cfg: dict):
    """Instantiate model from cfg['model'] section."""
    key = cfg["model"]["model"]
    if key not in MODEL_CLASSES:
        raise ValueError(f"Unknown model key: {key}")
    return MODEL_CLASSES[key](**cfg["model"]["para"])

def _load_first_pt(cfg: dict) -> str:
    """Return the first .pt file under {results_path}/model/."""
    mdir = os.path.join(cfg["paths"]["results_path"], "model")
    if not os.path.isdir(mdir):
        raise FileNotFoundError(f"Model dir not found: {mdir}")
    pts = [fn for fn in os.listdir(mdir) if fn.endswith(".pt")]
    if not pts:
        raise FileNotFoundError(f"No .pt weights in: {mdir}")
    return os.path.join(mdir, pts[0])

def _load_cfg(cfg_path: str) -> dict:
    with open(f"./configs/{cfg_path}", "r") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)

def _fmt_abs_tick(y, pos):
    """Formatter to show absolute value on y-axis."""
    return f"{int(abs(y))}"

def _save_c4s4_style(fig, dir, type, loc, name):
    """Save figure to {dir}/{loc}/{type}/{name}.png (C4S4-style)."""
    save_dir = os.path.join(dir, loc, type)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{name}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path

# ---------------------------- C5S1: original high-res input ----------------------------

def C5S1_fig01_original(
    Intep_cfgpath,                 # interpolation model cfg (not used here but kept for symmetry)
    MSFNO_cfgpath,                 # available if you choose msfno as predictor
    ResNet_cfgpath,                # available if you choose resnet as predictor
    predictor="resnet",            # {"resnet","msfno","mosrnet"}
    data_name="beamdi_num",        # fixed per request
    subset="beamdi_num_v1000",     # 1000 samples
    sample_index=34,               # pick the 34-th sample (0-based index)
    dir = r"./postprocessed",
    type = "fig",
    loc = "C5S1",
    name = "Original-input prediction vs Ground Truth"
):
    """
    Plot C5S1 (original data): predictor consumes high-res modes + grid channel.
    No baseline; model output is directly used. Ground Truth is drawn as band + step curve.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---- load cfgs ----
    Intep_cfg  = _load_cfg(Intep_cfgpath)   # kept for symmetry with reconstructed version
    MSFNO_cfg  = _load_cfg(MSFNO_cfgpath)
    ResNet_cfg = _load_cfg(ResNet_cfgpath)

    # choose predictor cfg
    pred_key = predictor.lower()
    if pred_key not in {"resnet","msfno","mosrnet"}:
        raise ValueError("predictor must be in {'resnet','msfno','mosrnet'}")
    if pred_key == "resnet":
        Predictor_cfg = ResNet_cfg
    elif pred_key == "msfno":
        Predictor_cfg = MSFNO_cfg
    else:  # "mosrnet"
        # If you keep a MoSRNet cfg separately, route it here; otherwise reuse whichever has "mosrnet"
        Predictor_cfg = MSFNO_cfg if MSFNO_cfg["model"]["model"] == "mosrnet" else ResNet_cfg

    # ---- build & load predictor ----
    predictor_model = _build_model_from_cfg(Predictor_cfg).to(device)
    pred_pt = _load_first_pt(Predictor_cfg)
    pred_state = torch.load(pred_pt, map_location=device, weights_only=True)
    predictor_model.load_state_dict(pred_state)
    predictor_model.eval()

    # ---- load dataset ----
    ds_path = f"./datasets/{data_name}/{subset}.pt"
    ds = torch.load(ds_path, map_location=device)
    # modes: [N, C(>=4), L]; dmg: [N, 1, L] or compatible
    in_ch = Predictor_cfg["train"]["in_channels"]
    modes = ds["mode"][:, :in_ch, :]
    dmg   = ds["dmg"].float()

    # ---- prepare one sample (original high-res) ----
    with torch.no_grad():
        x_hr = modes[sample_index:sample_index+1]             # [1, C, L]
        L = x_hr.shape[-1]
        grid = torch.linspace(0.0, 1.0, steps=L, device=device).reshape(1, 1, -1)
        x_in = torch.cat([x_hr.to(device), grid], dim=1)      # concat grid channel

        pred = predictor_model(x_in)                          # [1, 1, L] or [1, C, L]
        pred = 100 - pred.squeeze().detach().cpu().numpy() * 100

        gt   = 100 - dmg[sample_index].squeeze().detach().cpu().numpy() * 100

    # ---- plot ----
    x_idx  = np.linspace(0, 5400, num=pred.shape[-1])
    gt_idx = np.linspace(0, 5400, num=gt.shape[-1])

    pos_mask = pred > 0
    neg_mask = ~pos_mask

    fig, ax = plt.subplots(figsize=(10, 8))

    # prediction curve
    ax.plot(x_idx, pred, color="#3A3A3A", lw=1, alpha=1.0, zorder=3, label="_nolegend_")
    # fills
    ax.fill_between(x_idx, 0, pred, where=pos_mask, facecolor="#EE7E77", alpha=1.0, zorder=2, label="_nolegend_")
    ax.fill_between(x_idx, 0, pred, where=neg_mask, facecolor="#68A7BE", alpha=1.0, zorder=2, label="_nolegend_")

    # GT band + step
    ax.fill_between(gt_idx, 0, gt, facecolor="#CFCFCF", alpha=1.0, zorder=1, label="_nolegend_")
    ax.step(gt_idx, gt, color="#3A3A3A", lw=1, linestyle="--", alpha=1.0, where="mid", zorder=4, label="_nolegend_")

    # axes & cosmetics
    ax.axhline(0, color="k", linewidth=1.5, zorder=10)
    ax.set_xlim(0, 5400)
    ax.set_xticks(np.linspace(0, 5400, num=9))
    ax.set_ylim(-80, 80)
    ax.set_yticks(np.linspace(-75, 75, num=7))
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_abs_tick))

    ax.axhspan(0, 80,  color="#feece7", alpha=1.0, zorder=0)
    ax.axhspan(-80, 0, color="#deeeed", alpha=1.0, zorder=0)

    ax.text(150,  65, "Stiffness Loss",     color="#EE7E77", va="center", ha="left", fontsize=28, fontweight="bold")
    ax.text(150, -65, "Stiffness Increase", color="#68A7BE",  va="center", ha="left", fontsize=28, fontweight="bold")

    ax.set_ylabel("Stiffness Change (%)")
    ax.set_xlabel("Beam Span (mm)")

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
              labels=[f"{model_label}'s Prediction", "Ground Truth"],
              handler_map={tuple: HandlerTuple(ndivide=None)},
              loc="upper right")

    ax.grid(True, linestyle="-", alpha=0.3, zorder=0)
    plt.tight_layout()

    # save (append subset & sample index to name for traceability)
    if name is None:
        name = f"C5S1_Original_{pred_key.upper()}"
    name_full = f"{name}-{subset}-idx{sample_index}"
    return _save_c4s4_style(fig, dir, type, loc, name_full)

# ---------------------------- C5S1: downsample -> reconstruct -> predict ----------------------------

def C5S1_fig02_reconstructed(
    Intep_cfgpath,                 # used to reconstruct from downsampled input
    MSFNO_cfgpath,
    ResNet_cfgpath,
    predictor="resnet",
    data_name="beamdi_num",
    subset="beamdi_num_v1000",
    sample_index=34,
    down_indices=None,             # custom downsample index list; if None, use 9 points uniform-like
    dir = r"./postprocessed",
    type = "fig",
    loc = "C5S1",
    name = "Reconstructed-input prediction vs Ground Truth"
):
    """
    Plot C5S1 (reconstructed): downsample original modes -> reconstruction via Intep model -> predictor.
    No baseline; model output is directly used. Ground Truth is drawn as band + step curve.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---- load cfgs ----
    Intep_cfg  = _load_cfg(Intep_cfgpath)
    MSFNO_cfg  = _load_cfg(MSFNO_cfgpath)
    ResNet_cfg = _load_cfg(ResNet_cfgpath)

    # choose predictor cfg
    pred_key = predictor.lower()
    if pred_key not in {"resnet","msfno","mosrnet"}:
        raise ValueError("predictor must be in {'resnet','msfno','mosrnet'}")
    if pred_key == "resnet":
        Predictor_cfg = ResNet_cfg
    elif pred_key == "msfno":
        Predictor_cfg = MSFNO_cfg
    else:  # "mosrnet"
        Predictor_cfg = MSFNO_cfg if MSFNO_cfg["model"]["model"] == "mosrnet" else ResNet_cfg

    # ---- build & load models ----
    intep_model = _build_model_from_cfg(Intep_cfg).to(device)
    intep_pt = _load_first_pt(Intep_cfg)
    intep_state = torch.load(intep_pt, map_location=device, weights_only=True)
    intep_model.load_state_dict(intep_state)
    intep_model.eval()

    predictor_model = _build_model_from_cfg(Predictor_cfg).to(device)
    pred_pt = _load_first_pt(Predictor_cfg)
    pred_state = torch.load(pred_pt, map_location=device, weights_only=True)
    predictor_model.load_state_dict(pred_state)
    predictor_model.eval()

    # ---- load dataset ----
    ds_path = f"./datasets/{data_name}/{subset}.pt"
    ds = torch.load(ds_path, map_location=device)
    in_ch = Predictor_cfg["train"]["in_channels"]
    modes = ds["mode"][:, :in_ch, :]
    dmg   = ds["dmg"].float()

    # ---- define downsample indices ----
    L = modes.shape[-1]
    if down_indices is None:
        # default 9 points (similar to例子)：均匀到端点
        down_indices = np.linspace(0, L-1, num=9, dtype=int)

    # ---- reconstruct sample ----
    with torch.no_grad():
        x_hr = modes[sample_index:sample_index+1]                # [1, C, L]
        x_ds = x_hr[:, :, down_indices]                          # [1, C, K]
        x_rec = intep_model(x_ds.to(device))                     # [1, C', Lr] -> expect Lr ≈ L

        Lr = x_rec.shape[-1]
        grid = torch.linspace(0.0, 1.0, steps=Lr, device=device).reshape(1, 1, -1)
        x_in = torch.cat([x_rec, grid], dim=1)

        pred = predictor_model(x_in)                             # [1, 1, Lr] or [1, C, Lr]
        pred = 100 - pred.squeeze().detach().cpu().numpy() * 100

        gt   = 100 - dmg[sample_index].squeeze().detach().cpu().numpy() * 100

    # ---- plot ----
    x_idx  = np.linspace(0, 5400, num=pred.shape[-1])
    gt_idx = np.linspace(0, 5400, num=gt.shape[-1])

    pos_mask = pred > 0
    neg_mask = ~pos_mask

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(x_idx, pred, color="#3A3A3A", lw=1, alpha=1.0, zorder=3, label="_nolegend_")
    ax.fill_between(x_idx, 0, pred, where=pos_mask, facecolor="#EE7E77", alpha=1.0, zorder=2, label="_nolegend_")
    ax.fill_between(x_idx, 0, pred, where=neg_mask, facecolor="#68A7BE", alpha=1.0, zorder=2, label="_nolegend_")

    ax.fill_between(gt_idx, 0, gt, facecolor="#CFCFCF", alpha=1.0, zorder=1, label="_nolegend_")
    ax.step(gt_idx, gt, color="#3A3A3A", lw=1, linestyle="--", alpha=1.0, where="mid", zorder=4, label="_nolegend_")

    ax.axhline(0, color="k", linewidth=1.5, zorder=10)
    ax.set_xlim(0, 5400)
    ax.set_xticks(np.linspace(0, 5400, num=9))
    ax.set_ylim(-80, 80)
    ax.set_yticks(np.linspace(-75, 75, num=7))
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_abs_tick))

    ax.axhspan(0, 80,  color="#feece7", alpha=1.0, zorder=0)
    ax.axhspan(-80, 0, color="#deeeed", alpha=1.0, zorder=0)

    ax.text(150,  65, "Stiffness Loss",     color="#EE7E77", va="center", ha="left", fontsize=28, fontweight="bold")
    ax.text(150, -65, "Stiffness Increase", color="#68A7BE",  va="center", ha="left", fontsize=28, fontweight="bold")

    ax.set_ylabel("Stiffness Change (%)")
    ax.set_xlabel("Beam Span (mm)")

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
              labels=[f"{model_label}'s Prediction", "Ground Truth"],
              handler_map={tuple: HandlerTuple(ndivide=None)},
              loc="upper right")

    ax.grid(True, linestyle="-", alpha=0.3, zorder=0)
    plt.tight_layout()

    if name is None:
        name = f"C5S1_Reconstructed_{pred_key.upper()}"
    name_full = f"{name}-{subset}-idx{sample_index}"
    return _save_c4s4_style(fig, dir, type, loc, name_full)


# cfg paths
INTEP_CFG  = "mosrnet-beamdi_num_t8000-run01-250814-164957.yaml"
MSFNO_CFG  = "msfno-beamdi_num_t8000-run01-250814-165002.yaml"
RESNET_CFG = "resnet-beamdi_num_t8000-run01-250814-165006.yaml"
SAMPLE_INDEX = 34

# 原生数据预测
p1 = C5S1_fig01_original(
    Intep_cfgpath=INTEP_CFG,
    MSFNO_cfgpath=MSFNO_CFG,
    ResNet_cfgpath=RESNET_CFG,
    predictor="msfno",
    data_name="beamdi_num",
    subset="beamdi_num_v1000",
    sample_index=SAMPLE_INDEX,
    dir="./postprocessed",
    type="fig",
    loc="C5S1",
    name="numerical validation results _Ori_MSFNO"
)
print("Saved:", p1)

p2 = C5S1_fig01_original(
    Intep_cfgpath=INTEP_CFG,
    MSFNO_cfgpath=MSFNO_CFG,
    ResNet_cfgpath=RESNET_CFG,
    predictor="resnet",
    data_name="beamdi_num",
    subset="beamdi_num_v1000",
    sample_index=SAMPLE_INDEX,
    dir="./postprocessed",
    type="fig",
    loc="C5S1",
    name="numerical validation results _Ori_RESNET"
)
print("Saved:", p2)

# 下采样重建后的预测
p3 = C5S1_fig02_reconstructed(
    Intep_cfgpath=INTEP_CFG,
    MSFNO_cfgpath=MSFNO_CFG,
    ResNet_cfgpath=RESNET_CFG,
    predictor="msfno",
    data_name="beamdi_num",
    subset="beamdi_num_v1000",
    sample_index=SAMPLE_INDEX,
    down_indices=None, 
    dir="./postprocessed",
    type="fig",
    loc="C5S1",
    name="numerical validation results _Itp_MSFNO"
)
print("Saved:", p3)

p4 = C5S1_fig02_reconstructed(
    Intep_cfgpath=INTEP_CFG,
    MSFNO_cfgpath=MSFNO_CFG,
    ResNet_cfgpath=RESNET_CFG,
    predictor="resnet",
    data_name="beamdi_num",
    subset="beamdi_num_v1000",
    sample_index=SAMPLE_INDEX,
    down_indices=None,  
    dir="./postprocessed",
    type="fig",
    loc="C5S1",
    name="numerical validation results _Itp_RESNET"
)
print("Saved:", p4)