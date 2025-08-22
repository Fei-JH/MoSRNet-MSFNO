import os
import numpy as np
import torch
import yaml
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch as MplPatch
from matplotlib.legend_handler import HandlerTuple
from torch.utils.data import DataLoader, TensorDataset

from models.msfno import MSFNO
from models.resnet import ResNet
from models.mosrnet import MoSRNet
from utilities.euler_bernoulli_beam_fem import BeamAnalysis  # FEM utility

from losses.r2 import R2

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
# Function 1: cal r2 of the validation set before bias compensation
# =========================================================
def C5S2_r2_before_biascompen(
    Intep_cfgpath,                 # reconstruction/interpolation cfg (e.g., MoSRNet)
    MSFNO_cfgpath,                 # MSFNO predictor cfg
    ResNet_cfgpath, 
    predictor = "msfno",           # still needed to determine in_channels; not used for forward here
    data_name="beamdi_num",
    validdata="beamdi_num_v1000",
    down_idx=None,                 # if None -> 9-point uniform indices
    dir = r"./results/postprocessed",
    type = "csv",
    loc  = "C5S2",
    name = "C5S2-MS-FNO_SignedError+Intact"
):
    """
    One figure (MSFNO-centric):
      - Signed mean error (validation, MS-FNO) along span: mean over samples of (pred - GT).
      - Intact prediction curve produced by MS-FNO (after reconstruction).
    """

    # ----- load cfgs & models -----
    Intep_cfg  = _load_cfg(Intep_cfgpath)
    MSFNO_cfg  = _load_cfg(MSFNO_cfgpath)   # just to harmonize input channels
    ResNet_cfg = _load_cfg(ResNet_cfgpath)

    intep_model = _build_model_from_cfg(Intep_cfg).to(device)
    intep_model.load_state_dict(torch.load(_load_first_pt(Intep_cfg), map_location=device, weights_only=True))
    intep_model.eval()

    pred_key = predictor.lower()
    if pred_key not in {"resnet","msfno","mosrnet"}:
        raise ValueError("predictor must be in {'resnet','msfno','mosrnet'}")
    if pred_key == "resnet":
        Predictor_cfg = ResNet_cfg
    elif pred_key == "msfno":
        Predictor_cfg = MSFNO_cfg
    else:  # "mosrnet"
        raise KeyError ("the pred model must be msfno or resnet")

    pred_model = _build_model_from_cfg(Predictor_cfg).to(device)
    pred_model.load_state_dict(torch.load(_load_first_pt(Predictor_cfg), map_location=device, weights_only=True))
    pred_model.eval()

    # ----- dataset -----
    ds_path = f"./datasets/{data_name}/{validdata}.pt"
    ds = torch.load(ds_path, map_location=device)
    in_ch = max(3, MSFNO_cfg["train"]["in_channels"], ResNet_cfg["train"]["in_channels"])
    modes = ds["mode"][:, :in_ch, :]   # [N, C, L]
    dmg   = ds["dmg"].float()          # [N, 1, L]
    N, C, L = modes.shape

    if down_idx is None:
        down_idx = np.linspace(0, L-1, num=9, dtype=int)
    down_idx = np.asarray(down_idx, dtype=int)

    # ----- signed mean error (validation) -----
    X_down = modes[:, :, down_idx]
    Y_full = dmg
    loader = DataLoader(TensorDataset(X_down.float(), Y_full.float()),
                        batch_size=1, shuffle=False)

    evalr2 = R2()
    results = []

    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device)  # [B, C, K]
            y = y.to(device)  # [B, 1, L]

            x_rec = intep_model(x)  
            B, C_rec, Lr = x_rec.shape
            grid = torch.linspace(0.0, 1.0, steps=Lr, device=device).reshape(1,1,-1).expand(B,1,Lr)
            x_in = torch.cat([x_rec, grid], dim=1)   # [B, C_rec+1, Lr]

            # align GT length if needed
            if y.shape[-1] != Lr:
                y_aligned = torch.nn.functional.interpolate(y, size=Lr, mode="linear", align_corners=False)
            else:
                y_aligned = y

            y_pred = pred_model(x_in)
            r2loss = evalr2(y_pred.view(y_pred.shape[0], -1), y_aligned.view(y_aligned.shape[0], -1))
            results.append(r2loss.detach().cpu().item())

    # ---- compute mean ----
    meanr2 = np.mean(results)

    # ---- save as csv ----
    df = pd.DataFrame({
        "R2_each": results,
        "R2_mean": [meanr2]*len(results)   # 每行都填平均值，便于对齐
    })

    save_dir = os.path.join(dir, loc, type)
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{name}.csv")
    df.to_csv(csv_path, index=False)

    print(f"Saved results to {csv_path}")
    return df


# =========================================================
# Function 2: cal r2 of the validation set after bias compensation
# =========================================================
def C5S2_r2_after_biascompen(
    Intep_cfgpath,                 # reconstruction/interpolation cfg (e.g., MoSRNet)
    MSFNO_cfgpath,                 # MSFNO predictor cfg
    ResNet_cfgpath, 
    predictor = "msfno",           # still needed to determine in_channels; not used for forward here
    data_name="beamdi_num",
    validdata="beamdi_num_v1000",
    down_idx=None,                 # if None -> 9-point uniform indices
    dir = r"./results/postprocessed",
    L_beam=5.4, E=210e9, I=57.48e-8, rho=7850.0, A=65.423 * 0.0001,
    n_elem=540,
    type = "csv",
    loc  = "C5S2",
    name = "C5S2-MS-FNO_SignedError+Intact"
):
    """
    One figure (MSFNO-centric):
      - Signed mean error (validation, MS-FNO) along span: mean over samples of (pred - GT).
      - Intact prediction curve produced by MS-FNO (after reconstruction).
    """

    # ----- load cfgs & models -----
    Intep_cfg  = _load_cfg(Intep_cfgpath)
    MSFNO_cfg  = _load_cfg(MSFNO_cfgpath)   # just to harmonize input channels
    ResNet_cfg = _load_cfg(ResNet_cfgpath)

    intep_model = _build_model_from_cfg(Intep_cfg).to(device)
    intep_model.load_state_dict(torch.load(_load_first_pt(Intep_cfg), map_location=device, weights_only=True))
    intep_model.eval()

    pred_key = predictor.lower()
    if pred_key not in {"resnet","msfno","mosrnet"}:
        raise ValueError("predictor must be in {'resnet','msfno','mosrnet'}")
    if pred_key == "resnet":
        Predictor_cfg = ResNet_cfg
    elif pred_key == "msfno":
        Predictor_cfg = MSFNO_cfg
    else:  # "mosrnet"
        raise KeyError ("the pred model must be msfno or resnet")

    pred_model = _build_model_from_cfg(Predictor_cfg).to(device)
    pred_model.load_state_dict(torch.load(_load_first_pt(Predictor_cfg), map_location=device, weights_only=True))
    pred_model.eval()


    # ----- dataset -----
    ds_path = f"./datasets/{data_name}/{validdata}.pt"
    ds = torch.load(ds_path, map_location=device)
    in_ch = max(3, MSFNO_cfg["train"]["in_channels"], ResNet_cfg["train"]["in_channels"])
    modes = ds["mode"][:, :in_ch, :]   # [N, C, L]
    dmg   = ds["dmg"].float()          # [N, 1, L]
    N, C, L = modes.shape

    if down_idx is None:
        down_idx = np.linspace(0, L-1, num=9, dtype=int)
    down_idx = np.asarray(down_idx, dtype=int)

    # ----- signed mean error (validation) -----
    X_down = modes[:, :, down_idx]
    Y_full = dmg
    loader = DataLoader(TensorDataset(X_down.float(), Y_full.float()),
                        batch_size=1, shuffle=False)

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
    modes_down = torch.tensor(modes3[:, ds_idx], dtype=torch.float32, device=device).unsqueeze(0)
    print(modes3.shape) 
    print(modes_down.shape)  # [1,3,K]
    with torch.no_grad():
        modes_rec = intep_model(modes_down)  # [1,3,Lr_int]
    modes_rec = modes_rec.squeeze(0)         # [3,Lr_int]
    Lr_int = modes_rec.shape[-1]
    grid = torch.linspace(0.0, 1.0, steps=Lr_int, device=device).reshape(1,1,-1).squeeze(0)
    x_in_int = torch.cat([modes_rec, grid], dim=0).unsqueeze(0)  # [1,4,Lr_int]

    with torch.no_grad():
        pred_int_r = pred_model(x_in_int)

    evalr2 = R2()
    results = []

    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device)  # [B, C, K]
            y = y.to(device)  # [B, 1, L]

            x_rec = intep_model(x)  
            B, C_rec, Lr = x_rec.shape
            grid = torch.linspace(0.0, 1.0, steps=Lr, device=device).reshape(1,1,-1).expand(B,1,Lr)
            x_in = torch.cat([x_rec, grid], dim=1)   # [B, C_rec+1, Lr]

            # align GT length if needed
            if y.shape[-1] != Lr:
                y_aligned = torch.nn.functional.interpolate(y, size=Lr, mode="linear", align_corners=False)
            else:
                y_aligned = y
            
            y_pred = pred_int_r - pred_model(x_in)
            r2loss = evalr2(y_pred.view(y_pred.shape[0], -1), 1-y_aligned.view(y_aligned.shape[0], -1))
            results.append(r2loss.detach().cpu().item())

    # ---- compute mean ----
    meanr2 = np.mean(results)

    # ---- save as csv ----
    df = pd.DataFrame({
        "R2_each": results,
        "R2_mean": [meanr2]*len(results)   # 每行都填平均值，便于对齐
    })

    save_dir = os.path.join(dir, loc, type)
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{name}.csv")
    df.to_csv(csv_path, index=False)

    print(f"Saved results to {csv_path}")
    return df


# =========================================================
# Function 3: cal r2 of the validation set before bias compensation
# =========================================================
def C5S2_r2_onlypredmodel(
    Intep_cfgpath,                 # reconstruction/interpolation cfg (e.g., MoSRNet)
    MSFNO_cfgpath,                 # MSFNO predictor cfg
    ResNet_cfgpath, 
    predictor = "msfno",           # still needed to determine in_channels; not used for forward here
    data_name="beamdi_num",
    validdata="beamdi_num_v1000",
    down_idx=None,                 # if None -> 9-point uniform indices
    dir = r"./results/postprocessed",
    type = "csv",
    loc  = "C5S2",
    name = "C5S2-MS-FNO_SignedError+Intact"
):
    """
    One figure (MSFNO-centric):
      - Signed mean error (validation, MS-FNO) along span: mean over samples of (pred - GT).
      - Intact prediction curve produced by MS-FNO (after reconstruction).
    """

    # ----- load cfgs & models -----
    Intep_cfg  = _load_cfg(Intep_cfgpath)
    MSFNO_cfg  = _load_cfg(MSFNO_cfgpath)   # just to harmonize input channels
    ResNet_cfg = _load_cfg(ResNet_cfgpath)

    intep_model = _build_model_from_cfg(Intep_cfg).to(device)
    intep_model.load_state_dict(torch.load(_load_first_pt(Intep_cfg), map_location=device, weights_only=True))
    intep_model.eval()

    pred_key = predictor.lower()
    if pred_key not in {"resnet","msfno","mosrnet"}:
        raise ValueError("predictor must be in {'resnet','msfno','mosrnet'}")
    if pred_key == "resnet":
        Predictor_cfg = ResNet_cfg
    elif pred_key == "msfno":
        Predictor_cfg = MSFNO_cfg
    else:  # "mosrnet"
        raise KeyError ("the pred model must be msfno or resnet")

    pred_model = _build_model_from_cfg(Predictor_cfg).to(device)
    pred_model.load_state_dict(torch.load(_load_first_pt(Predictor_cfg), map_location=device, weights_only=True))
    pred_model.eval()

    # ----- dataset -----
    ds_path = f"./datasets/{data_name}/{validdata}.pt"
    ds = torch.load(ds_path, map_location=device)
    modes = ds["mode"][:, :, :]   # [N, C, L]
    dmg   = ds["dmg"].float()          # [N, 1, L]
    
    loader = DataLoader(TensorDataset(modes.float(), dmg.float()),
                        batch_size=1, shuffle=False)

    evalr2 = R2()
    results = []

    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device)  # [B, C, K]
            y = y.to(device)  # [B, 1, L]

            y_pred = pred_model(x)
            r2loss = evalr2(y_pred.view(y_pred.shape[0], -1), y.view(y.shape[0], -1))
            results.append(r2loss.detach().cpu().item())

    # ---- compute mean ----
    meanr2 = np.mean(results)

    # ---- save as csv ----
    df = pd.DataFrame({
        "R2_each": results,
        "R2_mean": [meanr2]*len(results)   # 每行都填平均值，便于对齐
    })

    save_dir = os.path.join(dir, loc, type)
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{name}.csv")
    df.to_csv(csv_path, index=False)

    print(f"Saved results to {csv_path}")
    return df

if __name__ == "__main__":
    # Example config paths (relative to ./configs/)
    INTEP_CFG  = "mosrnet-beamdi_num_t8000-run01-250814-164957.yaml"  # reconstruction/interpolation (e.g., MoSRNet)
    MSFNO_CFG  = "msfno-beamdi_num_t8000-run01-250814-165002.yaml"    # MSFNO predictor
    RESNET_CFG = "resnet-beamdi_num_t8000-run01-250814-165006.yaml"
    # ============================
    # 1. MS-FNO before bias compensation
    # ============================
    print(">>> Running MS-FNO R2 BEFORE bias compensation ...")
    df_msfno_before = C5S2_r2_before_biascompen(
        Intep_cfgpath=INTEP_CFG,
        MSFNO_cfgpath=MSFNO_CFG,
        ResNet_cfgpath=RESNET_CFG,
        predictor="msfno",
        data_name="beamdi_num",
        validdata="beamdi_num_v1000",
        loc="C5S2",
        name="C5S2-MS-FNO_r2_before"
    )
    print(df_msfno_before.head())

    # ============================
    # 2. MS-FNO after bias compensation
    # ============================
    print(">>> Running MS-FNO R2 AFTER bias compensation ...")
    df_msfno_after = C5S2_r2_after_biascompen(
        Intep_cfgpath=INTEP_CFG,
        MSFNO_cfgpath=MSFNO_CFG,
        ResNet_cfgpath=RESNET_CFG,
        predictor="msfno",
        data_name="beamdi_num",
        validdata="beamdi_num_v1000",
        loc="C5S2",
        name="C5S2-MS-FNO_r2_after"
    )
    print(df_msfno_after.head())

    # ============================
    # 3. MS-FNO original
    # ============================
    print(">>> Running MS-FNO R2 AFTER bias compensation ...")
    df_msfno_after = C5S2_r2_onlypredmodel(
        Intep_cfgpath=INTEP_CFG,
        MSFNO_cfgpath=MSFNO_CFG,
        ResNet_cfgpath=RESNET_CFG,
        predictor="msfno",
        data_name="beamdi_num",
        validdata="beamdi_num_v1000",
        loc="C5S2",
        name="C5S2-MS-FNO_r2_ori"
    )
    print(df_msfno_after.head())
    # ============================
    # 4. ResNet before bias compensation
    # ============================
    print(">>> Running ResNet R2 BEFORE bias compensation ...")
    df_resnet_before = C5S2_r2_before_biascompen(
        Intep_cfgpath=INTEP_CFG,
        MSFNO_cfgpath=MSFNO_CFG,
        ResNet_cfgpath=RESNET_CFG,
        predictor="resnet",
        data_name="beamdi_num",
        validdata="beamdi_num_v1000",
        loc="C5S2",
        name="C5S2-ResNet_r2_before"
    )
    print(df_resnet_before.head())

    # ============================
    # 5. ResNet after bias compensation
    # ============================
    print(">>> Running ResNet R2 AFTER bias compensation ...")
    df_resnet_after = C5S2_r2_after_biascompen(
        Intep_cfgpath=INTEP_CFG,
        MSFNO_cfgpath=MSFNO_CFG,
        ResNet_cfgpath=RESNET_CFG,
        predictor="resnet",
        data_name="beamdi_num",
        validdata="beamdi_num_v1000",
        loc="C5S2",
        name="C5S2-ResNet_r2_after"
    )
    print(df_resnet_after.head())

    # ============================
    # 6. ResNet original
    # ============================
    print(">>> Running MS-FNO R2 AFTER bias compensation ...")
    df_msfno_after = C5S2_r2_onlypredmodel(
        Intep_cfgpath=INTEP_CFG,
        MSFNO_cfgpath=MSFNO_CFG,
        ResNet_cfgpath=RESNET_CFG,
        predictor="resnet",
        data_name="beamdi_num",
        validdata="beamdi_num_v1000",
        loc="C5S2",
        name="C5S2-ResNet_r2_ori"
    )
    print(df_msfno_after.head())