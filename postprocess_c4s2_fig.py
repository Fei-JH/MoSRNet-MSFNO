'''
Author: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
Date: 2025-08-15 19:38:31
LastEditors: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
LastEditTime: 2025-10-21 15:41:38
'''

import os
import numpy as np
import torch
import yaml
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch as MplPatch
from matplotlib.legend_handler import HandlerTuple
from torch.utils.data import DataLoader, TensorDataset

from models.msfno import MSFNO
from models.resnet import ResNet
from models.mosrnet import MoSRNet
from utilities.euler_bernoulli_beam_fem import BeamAnalysis  # FEM utility

# ---------- Matplotlib global styles ----------
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 28
matplotlib.rcParams['axes.labelsize'] = 24
matplotlib.rcParams['legend.fontsize'] = 24
matplotlib.rcParams['axes.titlesize'] = 28
matplotlib.rcParams['xtick.labelsize'] = 24 
matplotlib.rcParams['ytick.labelsize'] = 24

# ---------- Globals & helpers ----------
MODEL_CLASSES = {"msfno": MSFNO, "resnet": ResNet, "mosrnet": MoSRNet}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def _load_cfg(cfg_path: str) -> dict:
    with open(f"./configs/{cfg_path}", "r") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)

def _build_model_from_cfg(cfg: dict):
    key = cfg["model"]["model"].lower()
    if key not in MODEL_CLASSES:
        raise ValueError(f"Unknown model key: {key}")
    return MODEL_CLASSES[key](**cfg["model"]["para"])

def _fmt_abs_tick(y, pos):
    """Formatter to show absolute value on y-axis."""
    return f"{int(abs(y))}"

def _load_first_pt(cfg: dict) -> str:
    mdir = os.path.join(cfg["paths"]["results_path"], "model")
    if not os.path.isdir(mdir):
        raise FileNotFoundError(f"Model dir not found: {mdir}")
    pts = [fn for fn in os.listdir(mdir) if fn.endswith(".pt")]
    if not pts:
        raise FileNotFoundError(f"No .pt found in: {mdir}")
    return os.path.join(mdir, pts[0])

def _save_fig(fig, dir, type, loc, name):
    save_dir = os.path.join(dir, loc, type)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{name}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path

# =========================================================
# Function 1: MSFNO-centric (signed mean error + intact prediction by MSFNO)
# =========================================================
def C4S2_MS_FNO_signed_error_plus_intact(
    Intep_cfgpath,                 # reconstruction/interpolation cfg (e.g., MoSRNet)
    MSFNO_cfgpath,                 # MSFNO predictor cfg
    ResNet_cfgpath,                # still needed to determine in_channels; not used for forward here
    data_name="beamdi_num",
    validdata="beamdi_num_v1000",
    down_idx=None,                 # if None -> 9-point uniform indices
    batch_size=32,
    # FEM (intact) parameters
    L_beam=5.4, E=210e9, I=57.48e-8, rho=7850.0, A=65.423 * 0.0001,
    n_elem=540,
    # save style
    dir = r"./results/postprocessed",
    type = "fig",
    loc  = "C4S2",
    name = "C4S2-MS-FNO_SignedError+Intact"
):
    """
    One figure (MSFNO-centric):
      - Signed mean error (validation, MS-FNO) along span: mean over samples of (pred - GT).
      - Intact prediction curve produced by MS-FNO (after reconstruction).
    """

    # ----- load cfgs & models -----
    Intep_cfg  = _load_cfg(Intep_cfgpath)
    MSFNO_cfg  = _load_cfg(MSFNO_cfgpath)
    ResNet_cfg = _load_cfg(ResNet_cfgpath)  # just to harmonize input channels

    intep_model = _build_model_from_cfg(Intep_cfg).to(device)
    intep_model.load_state_dict(torch.load(_load_first_pt(Intep_cfg), map_location=device, weights_only=True))
    intep_model.eval()

    msfno_model = _build_model_from_cfg(MSFNO_cfg).to(device)
    msfno_model.load_state_dict(torch.load(_load_first_pt(MSFNO_cfg), map_location=device, weights_only=True))
    msfno_model.eval()

    # ----- dataset -----
    ds_path = f"./datasets/{data_name}/{validdata}.pt"
    ds = torch.load(ds_path, map_location=device)
    # ensure channel count satisfies the predictor and is ≥ 4
    in_ch = max(4, MSFNO_cfg["train"]["in_channels"], ResNet_cfg["train"]["in_channels"])
    modes = ds["mode"][:, :in_ch, :]   # [N, C, L]
    dmg   = ds["dmg"].float()          # [N, 1, L]
    N, C, L = modes.shape

    if down_idx is None:
        down_idx = np.linspace(0, L-1, num=9, dtype=int)
    down_idx = np.asarray(down_idx, dtype=int)

    # ----- signed mean error (validation) for MSFNO -----
    X_down = modes[:, :, down_idx]
    Y_full = dmg
    loader = DataLoader(TensorDataset(X_down.float(), Y_full.float()),
                        batch_size=batch_size, shuffle=False)

    err_sum = None
    total = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)  # [B, C, K]
            yb = yb.to(device)  # [B, 1, L]

            x_rec = intep_model(xb)                  # [B, C_rec, Lr]
            B, C_rec, Lr = x_rec.shape
            grid = torch.linspace(0.0, 1.0, steps=Lr, device=device).reshape(1,1,-1).expand(B,1,Lr)
            x_in = torch.cat([x_rec, grid], dim=1)   # [B, C_rec+1, Lr]

            # align GT length if needed
            if yb.shape[-1] != Lr:
                y_aligned = torch.nn.functional.interpolate(yb, size=Lr, mode="linear", align_corners=False)
            else:
                y_aligned = yb

            gt_pct  = (100 - y_aligned.squeeze(1) * 100).detach().cpu().numpy()           # [B, Lr]
            pred_ms = (100 - msfno_model(x_in).squeeze(1) * 100).detach().cpu().numpy()   # [B, Lr]

            # SIGNED error: pred - GT, then sum over batch
            err = pred_ms.squeeze() - gt_pct    # [B, Lr]
            batch_sum = err.sum(axis=0)                            # [Lr]

            if err_sum is None:
                err_sum = batch_sum
            else:
                if err_sum.shape[0] != batch_sum.shape[0]:
                    # defensive length alignment (rare)
                    x_old = np.linspace(0, 1, num=batch_sum.shape[0])
                    x_new = np.linspace(0, 1, num=err_sum.shape[0])
                    batch_sum = np.interp(x_new, x_old, batch_sum)
                err_sum += batch_sum

            total += B

    signed_mean_err = err_sum / total                    # [L*]
    x_mm_err = np.linspace(0, 5400, num=signed_mean_err.shape[0])

    # ----- intact FEM(1~3) → downsample → reconstruct → predictor (MSFNO only) -----
    intact = np.ones(n_elem)
    beam   = BeamAnalysis(L_beam, E, I, rho, A, n_elem)
    beam.assemble_matrices(dmgfield=intact, mass_dmg_power=0)
    beam.apply_BC()
    _, eigenvectors = beam.solve_eigenproblem()
    u_vectors, _    = beam.split_eigenvectors(eigenvectors)
    modes3 = u_vectors[:, 2:5].T  # [3, n_elem]

    # sign align + max-abs normalize
    for i in range(modes3.shape[0]):
        if modes3[i, 14] < 0:
            modes3[i] = -modes3[i]
        ma = np.max(np.abs(modes3[i]))
        if ma > 0:
            modes3[i] = modes3[i] / ma

    ds_idx = down_idx
    modes_down = torch.tensor(modes3[:, ds_idx], dtype=torch.float32, device=device).unsqueeze(0)  # [1,3,K]
    with torch.no_grad():
        modes_rec = intep_model(modes_down)  # [1,3,Lr_int]
    modes_rec = modes_rec.squeeze(0)         # [3,Lr_int]
    Lr_int = modes_rec.shape[-1]
    grid = torch.linspace(0.0, 1.0, steps=Lr_int, device=device).reshape(1,1,-1).squeeze(0)
    x_in_int = torch.cat([modes_rec, grid], dim=0).unsqueeze(0)  # [1,4,Lr_int]

    with torch.no_grad():
        pred_int_m = (100 - msfno_model(x_in_int).squeeze() * 100).detach().cpu().numpy()

    x_mm_int = np.linspace(0, 5400, num=pred_int_m.shape[-1])

    # ----- plot (MSFNO-centric) -----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_mm_err, signed_mean_err, lw=2, color="#d62728", linestyle="-", label="Mean error on the validation set (MoSRNet + MS-FNO)")
    ax.plot(x_mm_int, pred_int_m,     lw=2, color="#1f77b4", linestyle="--", label="Prediction of intact scenario (MoSRNet + MS-FNO)")

    ax.axhline(0.0, color="k", linewidth=1.2)
    ax.set_xlim(0, 5400)
    ax.set_xticks(np.linspace(0, 5400, num=9))
    ax.set_ylim(-9, 21)
    ax.set_yticks(np.linspace(-8, 20, num=8))
    ax.set_xlabel("Beam Span (mm)")
    ax.set_ylabel("Stiffness Loss (%)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper center", fontsize=20)
    plt.tight_layout()

    return _save_fig(fig, dir, type, loc, name)

# =========================================================
# Function 2: ResNet-centric (signed mean error + intact prediction by ResNet)
# =========================================================
def C4S2_ResNet_signed_error_plus_intact(
    Intep_cfgpath,                 # reconstruction/interpolation cfg (e.g., MoSRNet)
    MSFNO_cfgpath,                 # still needed to determine in_channels; not used for forward here
    ResNet_cfgpath,                # ResNet predictor cfg
    data_name="beamdi_num",
    validdata="beamdi_num_v1000",
    down_idx=None,                 # if None -> 9-point uniform indices
    batch_size=32,
    # FEM (intact) parameters
    L_beam=5.4, E=210e9, I=57.48e-8, rho=7850.0, A=65.423 * 0.0001,
    n_elem=540,
    # save style
    dir = r"./results/postprocessed",
    type = "fig",
    loc  = "C4S2",
    name = "C4S2-ResNet_SignedError+Intact"
):
    """
    One figure (ResNet-centric):
      - Signed mean error (validation, ResNet) along span: mean over samples of (pred - GT).
      - Intact prediction curve produced by ResNet (after reconstruction).
    """

    # ----- load cfgs & models -----
    Intep_cfg  = _load_cfg(Intep_cfgpath)
    MSFNO_cfg  = _load_cfg(MSFNO_cfgpath)   # just to harmonize input channels
    ResNet_cfg = _load_cfg(ResNet_cfgpath)

    intep_model = _build_model_from_cfg(Intep_cfg).to(device)
    intep_model.load_state_dict(torch.load(_load_first_pt(Intep_cfg), map_location=device, weights_only=True))
    intep_model.eval()

    resnet_model = _build_model_from_cfg(ResNet_cfg).to(device)
    resnet_model.load_state_dict(torch.load(_load_first_pt(ResNet_cfg), map_location=device, weights_only=True))
    resnet_model.eval()

    # ----- dataset -----
    ds_path = f"./datasets/{data_name}/{validdata}.pt"
    ds = torch.load(ds_path, map_location=device)
    in_ch = max(4, MSFNO_cfg["train"]["in_channels"], ResNet_cfg["train"]["in_channels"])
    modes = ds["mode"][:, :in_ch, :]   # [N, C, L]
    dmg   = ds["dmg"].float()          # [N, 1, L]
    N, C, L = modes.shape

    if down_idx is None:
        down_idx = np.linspace(0, L-1, num=9, dtype=int)
    down_idx = np.asarray(down_idx, dtype=int)

    # ----- signed mean error (validation) for ResNet -----
    X_down = modes[:, :, down_idx]
    Y_full = dmg
    loader = DataLoader(TensorDataset(X_down.float(), Y_full.float()),
                        batch_size=batch_size, shuffle=False)

    err_sum = None
    total = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)  # [B, C, K]
            yb = yb.to(device)  # [B, 1, L]

            x_rec = intep_model(xb)                  # [B, C_rec, Lr]
            B, C_rec, Lr = x_rec.shape
            grid = torch.linspace(0.0, 1.0, steps=Lr, device=device).reshape(1,1,-1).expand(B,1,Lr)
            x_in = torch.cat([x_rec, grid], dim=1)   # [B, C_rec+1, Lr]

            # align GT length if needed
            if yb.shape[-1] != Lr:
                y_aligned = torch.nn.functional.interpolate(yb, size=Lr, mode="linear", align_corners=False)
            else:
                y_aligned = yb

            gt_pct  = (100 - y_aligned.squeeze(1) * 100).detach().cpu().numpy()             # [B, Lr]
            pred_rs = (100 - resnet_model(x_in).squeeze(1) * 100).detach().cpu().numpy()   # [B, Lr]

            # SIGNED error: pred - GT
            err = pred_rs.squeeze() - gt_pct      # [B, Lr]
            batch_sum = err.sum(axis=0)                             # [Lr]

            if err_sum is None:
                err_sum = batch_sum
            else:
                if err_sum.shape[0] != batch_sum.shape[0]:
                    x_old = np.linspace(0, 1, num=batch_sum.shape[0])
                    x_new = np.linspace(0, 1, num=err_sum.shape[0])
                    batch_sum = np.interp(x_new, x_old, batch_sum)
                err_sum += batch_sum

            total += B

    signed_mean_err = err_sum / total                    # [L*]
    x_mm_err = np.linspace(0, 5400, num=signed_mean_err.shape[0])

    # ----- intact FEM(1~3) → downsample → reconstruct → predictor (ResNet only) -----
    intact = np.ones(n_elem)
    beam   = BeamAnalysis(L_beam, E, I, rho, A, n_elem)
    beam.assemble_matrices(dmgfield=intact, mass_dmg_power=0)
    beam.apply_BC()
    _, eigenvectors = beam.solve_eigenproblem()
    u_vectors, _    = beam.split_eigenvectors(eigenvectors)
    modes3 = u_vectors[:, 2:5].T  # [3, n_elem]

    # sign align + max-abs normalize
    for i in range(modes3.shape[0]):
        if modes3[i, 14] < 0:
            modes3[i] = -modes3[i]
        ma = np.max(np.abs(modes3[i]))
        if ma > 0:
            modes3[i] = modes3[i] / ma

    ds_idx = down_idx
    modes_down = torch.tensor(modes3[:, ds_idx], dtype=torch.float32, device=device).unsqueeze(0)  # [1,3,K]
    with torch.no_grad():
        modes_rec = intep_model(modes_down)  # [1,3,Lr_int]
    modes_rec = modes_rec.squeeze(0)         # [3,Lr_int]
    Lr_int = modes_rec.shape[-1]
    grid = torch.linspace(0.0, 1.0, steps=Lr_int, device=device).reshape(1,1,-1).squeeze(0)
    x_in_int = torch.cat([modes_rec, grid], dim=0).unsqueeze(0)  # [1,4,Lr_int]

    with torch.no_grad():
        pred_int_r = (100 - resnet_model(x_in_int).squeeze() * 100).detach().cpu().numpy()

    x_mm_int = np.linspace(0, 5400, num=pred_int_r.shape[-1])

    # ----- plot (ResNet-centric) -----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_mm_err, signed_mean_err, lw=2, color="#d62728", linestyle="-", label="Mean error on the validation set (MoSRNet + ResNet)")
    ax.plot(x_mm_int, pred_int_r,     lw=2, color="#1f77b4", linestyle="--", label="Prediction of intact scenario (MoSRNet + ResNet)")

    ax.axhline(0.0, color="k", linewidth=1.2)
    ax.set_xlim(0, 5400)
    ax.set_xticks(np.linspace(0, 5400, num=9))
    ax.set_ylim(-30, 90)
    ax.set_yticks(np.linspace(-20, 80, num=6))
    ax.set_xlabel("Beam Span (mm)")
    ax.set_ylabel("Stiffness Loss (%)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper center", fontsize=20)
    plt.tight_layout()

    return _save_fig(fig, dir, type, loc, name)


def C4S2_reconstructed_withbaseline(
    Intep_cfgpath,                 # used to reconstruct from downsampled input
    MSFNO_cfgpath,
    ResNet_cfgpath,
    predictor="resnet",
    data_name="beamdi_num",
    subset="beamdi_num_v1000",
    sample_index=34,
    down_idx=None,             # custom downsample index list; if None, use 9 points uniform-like
    dir = r"./results/postprocessed",
    L_beam=5.4, E=210e9, I=57.48e-8, rho=7850.0, A=65.423 * 0.0001,
    n_elem=540,
    type = "fig",
    loc = "C4S2",
    name = "Reconstructed-input prediction vs Ground Truth after bias compensation"
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
    if down_idx is None:
        # default 9 points (similar to例子)：均匀到端点
        down_idx = np.linspace(0, L-1, num=9, dtype=int)

    # ---- reconstruct sample ----
    with torch.no_grad():
        x_hr = modes[sample_index:sample_index+1]                # [1, C, L]
        x_ds = x_hr[:, :, down_idx]                          # [1, C, K]
        x_rec = intep_model(x_ds.to(device))                     # [1, C', Lr] -> expect Lr ≈ L

        Lr = x_rec.shape[-1]
        grid = torch.linspace(0.0, 1.0, steps=Lr, device=device).reshape(1, 1, -1)
        x_in = torch.cat([x_rec, grid], dim=1)

        pred = predictor_model(x_in)                             # [1, 1, Lr] or [1, C, Lr]
        pred = 100 - pred.squeeze().detach().cpu().numpy() * 100

        gt   = 100 - dmg[sample_index].squeeze().detach().cpu().numpy() * 100

        intact = np.ones(n_elem)
    beam   = BeamAnalysis(L_beam, E, I, rho, A, n_elem)
    beam.assemble_matrices(dmgfield=intact, mass_dmg_power=0)
    beam.apply_BC()
    _, eigenvectors = beam.solve_eigenproblem()
    u_vectors, _    = beam.split_eigenvectors(eigenvectors)
    modes3 = u_vectors[:, 2:5].T  # [3, n_elem]

    # sign align + max-abs normalize
    for i in range(modes3.shape[0]):
        if modes3[i, 14] < 0:
            modes3[i] = -modes3[i]
        ma = np.max(np.abs(modes3[i]))
        if ma > 0:
            modes3[i] = modes3[i] / ma

    ds_idx = down_idx
    modes_down = torch.tensor(modes3[:, ds_idx], dtype=torch.float32, device=device).unsqueeze(0)  # [1,3,K]
    with torch.no_grad():
        modes_rec = intep_model(modes_down)  # [1,3,Lr_int]
    modes_rec = modes_rec.squeeze(0)         # [3,Lr_int]
    Lr_int = modes_rec.shape[-1]
    grid = torch.linspace(0.0, 1.0, steps=Lr_int, device=device).reshape(1,1,-1).squeeze(0)
    x_in_int = torch.cat([modes_rec, grid], dim=0).unsqueeze(0)  # [1,4,Lr_int]

    with torch.no_grad():
        pred_int_r = (100 - predictor_model(x_in_int).squeeze() * 100).detach().cpu().numpy()

    # ---- plot ----
    x_idx  = np.linspace(0, 5400, num=pred.shape[-1])
    gt_idx = np.linspace(0, 5400, num=gt.shape[-1])

    pred = pred - pred_int_r

    pos_mask = pred > 0
    neg_mask = ~pos_mask

    fig, ax = plt.subplots(figsize=(10, 7.5))

    ax.plot(x_idx, pred, color="#3A3A3A", lw=1, alpha=1.0, zorder=3, label="_nolegend_")
    ax.fill_between(x_idx, 0, pred, where=pos_mask, facecolor="#EE7E77", alpha=1.0, zorder=2, label="_nolegend_")
    ax.fill_between(x_idx, 0, pred, where=neg_mask, facecolor="#68A7BE", alpha=1.0, zorder=2, label="_nolegend_")

    ax.fill_between(gt_idx, 0, gt, facecolor="#CFCFCF", alpha=1.0, zorder=1, label="_nolegend_")
    ax.step(gt_idx, gt, color="#3A3A3A", lw=1, linestyle="--", alpha=1.0, where="mid", zorder=4, label="_nolegend_")

    ax.axhline(0, color="k", linewidth=1.5, zorder=10)
    ax.set_xlim(0, 5400)
    ax.set_xticks(np.linspace(0, 5400, num=9))
    ax.set_ylim(-85, 85)
    ax.set_yticks(np.linspace(-75, 75, num=7))
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_abs_tick))

    ax.axhspan(0, 85,  color="#feece7", alpha=1.0, zorder=0)
    ax.axhspan(-85, 0, color="#deeeed", alpha=1.0, zorder=0)

    ax.text(150,  65, "Stiffness Loss",     color="#EE7E77", va="center", ha="left", fontsize=28, fontweight="bold")
    ax.text(150, -65, "Stiffness Increase", color="#68A7BE",  va="center", ha="left", fontsize=28, fontweight="bold")

    ax.set_ylabel("Stiffness Change (%)")
    ax.set_xlabel("Beam Span (mm)")

    red_patch  = MplPatch(color="#EE7E77")
    blue_patch = MplPatch(color="#68A7BE")
    gt_patch   = MplPatch(facecolor="#CFCFCF")

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
    return _save_fig(fig, dir, type, loc, name_full)


if __name__ == "__main__":
    """
    Execute C4S2: generate four figures
      1) Original:   MoSRNet + MS-FNO signed mean error + intact baseline (reference curve)
      2) Original:   MoSRNet + ResNet signed mean error + intact baseline (reference curve)
      3) New:        Baseline-corrected mean error (MoSRNet + MS-FNO)
      4) New:        Baseline-corrected mean error (MoSRNet + ResNet)

    Edit the three CFG file names below to your actual config filenames.
    All figures will be saved under: ./postprocessed/C4S2/fig/
    """
    # --------- user-configurable paths (EDIT to your actual cfg names) ---------
    INTEP_CFG  = "mosrnet-beamdi_num_t8000-run01-250814-164957.yaml"  # reconstruction/interpolation (e.g., MoSRNet)
    MSFNO_CFG  = "msfno-beamdi_num_t8000-run01-250814-165002.yaml"    # MSFNO predictor
    RESNET_CFG = "resnet-beamdi_num_t8000-run01-250814-165006.yaml"   # ResNet predictor
    SAMPLE_INDEX = 34

    # --------- dataset params ---------
    data_name = "beamdi_num"
    validdata = "beamdi_num_v1000"

    # --------- downsample indices (9 points including both ends) ---------
    down_idx = np.array([0, 68, 135, 203, 270, 337, 405, 473, 540], dtype=int)

    # --------- save style ---------
    out_dir  = "./results/postprocessed"
    out_type = "fig"
    out_loc  = "C4S2"

    # =================== 1) Original (MS-FNO) ===================
    out1 = C4S2_MS_FNO_signed_error_plus_intact(
        Intep_cfgpath=INTEP_CFG,
        MSFNO_cfgpath=MSFNO_CFG,
        ResNet_cfgpath=RESNET_CFG,      # only used to harmonize in_channels
        data_name=data_name,
        validdata=validdata,
        down_idx=down_idx,
        batch_size=32,
        dir=out_dir, type=out_type, loc=out_loc,
        name="C4S2-MS-FNO_SignedError+Intact"
    )
    print("[C4S2] Saved:", out1)

    # =================== 2) Original (ResNet) ===================
    out2 = C4S2_ResNet_signed_error_plus_intact(
        Intep_cfgpath=INTEP_CFG,
        MSFNO_cfgpath=MSFNO_CFG,        # only used to harmonize in_channels
        ResNet_cfgpath=RESNET_CFG,
        data_name=data_name,
        validdata=validdata,
        down_idx=down_idx,
        batch_size=32,
        dir=out_dir, type=out_type, loc=out_loc,
        name="C4S2-ResNet_SignedError+Intact"
    )
    print("[C4S2] Saved:", out2)

    # 下采样重建后的预测
    p3 = C4S2_reconstructed_withbaseline(
        Intep_cfgpath=INTEP_CFG,
        MSFNO_cfgpath=MSFNO_CFG,
        ResNet_cfgpath=RESNET_CFG,
        predictor="msfno",
        data_name="beamdi_num",
        subset="beamdi_num_v1000",
        sample_index=SAMPLE_INDEX,
        down_idx=None, 
        dir="./results/postprocessed",
        L_beam=5.4, E=210e9, I=57.48e-8, rho=7850.0, A=65.423 * 0.0001,
        n_elem=540,
        type="fig",
        loc="C4S2",
        name="numerical validation results after bias compensation_Itp_MSFNO"
    )
    print("Saved:", p3)

    p4 = C4S2_reconstructed_withbaseline(
        Intep_cfgpath=INTEP_CFG,
        MSFNO_cfgpath=MSFNO_CFG,
        ResNet_cfgpath=RESNET_CFG, 
        predictor="resnet",
        data_name="beamdi_num",
        subset="beamdi_num_v1000",
        sample_index=SAMPLE_INDEX,
        down_idx=None,  
        dir="./results/postprocessed",
        L_beam=5.4, E=210e9, I=57.48e-8, rho=7850.0, A=65.423 * 0.0001,
        n_elem=540,
        type="fig",
        loc="C4S2",
        name="numerical validation results after bias compensation_Itp_RESNET"
    )
    print("Saved:", p4)

