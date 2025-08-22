import os
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

from models.msfno import MSFNO
from models.resnet import ResNet
from losses.r2 import R2

# ----------------------- Globals -----------------------
MODEL_CLASSES = {"msfno": MSFNO, "resnet": ResNet}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# ----------------------- Utilities -----------------------
def _load_cfg(cfg_path: str) -> dict:
    """Load YAML cfg under ./configs/"""
    with open(os.path.join("./configs", cfg_path), "r") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)

def _build_model_from_cfg(cfg: dict):
    """Instantiate model from cfg."""
    key = cfg["model"]["model"].lower()
    if key not in MODEL_CLASSES:
        raise ValueError(f"Unknown model key: {key}")
    return MODEL_CLASSES[key](**cfg["model"]["para"])

def _load_first_pt(cfg: dict) -> str:
    """Find first .pt weight under results_path/model."""
    mdir = os.path.join(cfg["paths"]["results_path"], "model")
    if not os.path.isdir(mdir):
        raise FileNotFoundError(f"Model dir not found: {mdir}")
    pts = [fn for fn in os.listdir(mdir) if fn.endswith(".pt")]
    if not pts:
        raise FileNotFoundError(f"No .pt found in: {mdir}")
    pts.sort()
    return os.path.join(mdir, pts[0])

def _rms_along_len(x: torch.Tensor) -> torch.Tensor:
    """Compute RMS along the last dimension (length axis). Shape preserved except last dim."""
    # x: [B, C, L] -> rms over L => [B, C, 1]
    return torch.sqrt(torch.clamp((x ** 2).mean(dim=-1, keepdim=True), min=1e-30))

def _add_rms_white_noise(x: torch.Tensor, pct: float) -> torch.Tensor:
    """
    Add zero-mean white noise with target RMS = pct * RMS(signal) along last dim.
    The noise is normalized per (B,C) to unit RMS before scaling to the exact target.
    Compatible with older PyTorch versions (no generator arg).
    """
    if pct <= 0:
        return x
    n = torch.randn(x.shape, device=x.device)        # Gaussian white noise
    n_rms = _rms_along_len(n)                        # [B,C,1]
    n_norm = n / n_rms                               # unit-RMS noise along length
    sig_rms = _rms_along_len(x)                      # [B,C,1]
    target = pct * sig_rms                           # [B,C,1]
    return x + n_norm * target

def _forward_predict(model, x_modes: torch.Tensor) -> torch.Tensor:
    """
    Forward pass helper.
    - x_modes: [B, C, L]
    - Append a normalized grid channel to match (C_in+1) interface if needed.
    - Return: y_pred [B, 1, L_out]
    """
    B, C, L = x_modes.shape
    grid = torch.linspace(0.0, 1.0, steps=L, device=x_modes.device).reshape(1, 1, L).expand(B, 1, L)
    x_in = torch.cat([x_modes, grid], dim=1)  # [B, C+1, L]
    with torch.no_grad():
        y_pred = model(x_in)
    return y_pred

def _compute_dataset_r2(model, loader, L_target: int, evalr2: R2) -> float:
    """
    Compute mean R^2 over the entire validation set.
    - Model outputs may have L_out != L_target (GT length). Interpolate to L_target if needed.
    - Returns mean R^2 across samples.
    """
    r2_list = []
    with torch.no_grad():
        for x_modes, y_gt in loader:
            x_modes = x_modes.to(device)  # [B, C, L]
            y_gt    = y_gt.to(device)     # [B, 1, L_target]

            y_pred = _forward_predict(model, x_modes)      # [B, 1, L_out]
            B, _, L_out = y_pred.shape

            if L_out != L_target:
                # Align prediction to GT length
                y_pred_aligned = torch.nn.functional.interpolate(y_pred, size=L_target, mode="linear", align_corners=False)
            else:
                y_pred_aligned = y_pred

            # Flatten length dimension for R^2
            r2loss = evalr2(y_pred_aligned.view(B, -1), y_gt.view(B, -1))
            r2_list.append(float(r2loss.detach().cpu().item()))
    return float(np.mean(r2_list)) if len(r2_list) > 0 else float("nan")


# ----------------------- Main Plot Function -----------------------
def C5S4_noise_sensitivity_direct_plot(
    MSFNO_cfgpath,                 # MSFNO cfg
    ResNet_cfgpath,                # ResNet cfg
    data_name="beamdi_num",
    validdata="beamdi_num_v1000",
    dir  = r"./results/postprocessed",
    type = "fig",
    loc  = "C5S4",
    name = "R2_vs_RMS_WhiteNoise_MSFNO_vs_ResNet",
    noise_max_pct = 0.10,          # 10% RMS noise
    noise_step    = 0.005,         # 0.5% step
    batch_size    = 4,
    seed          = 123,
    show          = True,
    save_fig      = True,
    save_csv      = True,
):
    """
    Evaluate MSFNO and ResNet directly on original modal data with RMS white noise added.
    - No downsampling or reconstruction.
    - No intact baseline subtraction (no bias compensation).
    - Scan noise levels from 0% to 10% (step 0.5%) and compute mean R^2 on the validation set.
    - Plot both curves in a single figure and save CSV.
    """

    # ---- Reproducibility (CPU & CUDA, best-effort) ----
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ---- Load cfgs & models ----
    MSFNO_cfg  = _load_cfg(MSFNO_cfgpath)
    ResNet_cfg = _load_cfg(ResNet_cfgpath)

    msfno_model = _build_model_from_cfg(MSFNO_cfg).to(device)
    msfno_model.load_state_dict(torch.load(_load_first_pt(MSFNO_cfg), map_location=device, weights_only=True))
    msfno_model.eval()

    resnet_model = _build_model_from_cfg(ResNet_cfg).to(device)
    resnet_model.load_state_dict(torch.load(_load_first_pt(ResNet_cfg), map_location=device, weights_only=True))
    resnet_model.eval()

    # ---- Dataset ----
    ds_path = os.path.join("./datasets", data_name, f"{validdata}.pt")
    ds = torch.load(ds_path, map_location=device)
    modes_full = ds["mode"].float()  # [N, C_all, L]
    dmg_full   = ds["dmg"].float()   # [N, 1, L]
    N, C_all, L = modes_full.shape

    # in_channels for each model
    in_ch_msfno  = int(MSFNO_cfg["train"]["in_channels"])
    in_ch_resnet = int(ResNet_cfg["train"]["in_channels"])

    # Build two DataLoaders (we will replace tensors with noisy versions per noise level)
    # For memory efficiency, we keep CPU tensors in DataLoader and move to device in the loop.
    # Here we prepare base (no-noise) tensors; noise will be applied on the fly.
    X_base_msfno  = modes_full[:, :in_ch_msfno, :].cpu()
    X_base_resnet = modes_full[:, :in_ch_resnet, :].cpu()
    Y_full_cpu    = dmg_full.cpu()

    # Create placeholder datasets; actual x will be replaced after adding noise
    # We pass the base X to satisfy Dataset shape; inside the loop we ignore it and use noisy x.
    ds_msfno  = TensorDataset(X_base_msfno, Y_full_cpu)
    ds_resnet = TensorDataset(X_base_resnet, Y_full_cpu)

    loader_msfno  = DataLoader(ds_msfno,  batch_size=batch_size, shuffle=False)
    loader_resnet = DataLoader(ds_resnet, batch_size=batch_size, shuffle=False)

    # ---- Noise scan settings ----
    noise_pcts = np.round(np.arange(0.0, noise_max_pct + 1e-12, noise_step), 6)
    r2_msfno_list  = []
    r2_resnet_list = []

    evalr2 = R2()

    # ---- Scan noise levels ----
    for pct in noise_pcts:
        # Create brand-new noisy copies (on device) for this pct
        # Note: we add noise per-batch inside the loop for memory friendliness.
        # To keep the logic simple, we construct new loaders that yield noisy x on-the-fly.

        def _yield_noisy_batches(loader, in_ch: int):
            """Generator that yields (x_noisy, y) on device with given in_channels."""
            for xb, yb in loader:
                xb = xb.to(device)  # [B, in_ch, L]
                yb = yb.to(device)  # [B, 1, L]
                xb_noisy = _add_rms_white_noise(xb, pct)  # add RMS-based white noise
                yield xb_noisy, yb

        # Evaluate MSFNO
        r2_list_msfno = []
        with torch.no_grad():
            for x_noisy, y in _yield_noisy_batches(loader_msfno, in_ch_msfno):
                y_pred = _forward_predict(msfno_model, x_noisy)  # [B,1,L_out]
                B, _, L_out = y_pred.shape
                r2loss = evalr2(y_pred.view(B, -1), y.view(B, -1))
                r2_list_msfno.append(float(r2loss.detach().cpu().item()))
        r2_msfno = float(np.mean(r2_list_msfno)) if len(r2_list_msfno) > 0 else float("nan")
        r2_msfno_list.append(r2_msfno)

        # Evaluate ResNet
        r2_list_resnet = []
        with torch.no_grad():
            for x_noisy, y in _yield_noisy_batches(loader_resnet, in_ch_resnet):
                y_pred = _forward_predict(resnet_model, x_noisy)  # [B,1,L_out]
                B, _, L_out = y_pred.shape
                r2loss = evalr2(y_pred.view(B, -1), y.view(B, -1))
                r2_list_resnet.append(float(r2loss.detach().cpu().item()))
        r2_resnet = float(np.mean(r2_list_resnet)) if len(r2_list_resnet) > 0 else float("nan")
        r2_resnet_list.append(r2_resnet)

        print(f"[Noise {pct*100:.2f}%] R2 - MSFNO: {r2_msfno:.4f} | ResNet: {r2_resnet:.4f}")

    # ---- Save CSV ----
    save_dir = os.path.join(dir, loc, type)
    os.makedirs(save_dir, exist_ok=True)

    df = pd.DataFrame({
        "noise_pct": (noise_pcts * 100.0),
        "R2_MSFNO": r2_msfno_list,
        "R2_ResNet": r2_resnet_list
    })
    if save_csv:
        csv_path = os.path.join(save_dir, f"{name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Saved CSV to: {csv_path}")

    # ---- Plot both curves ----
    plt.figure(figsize=(7.0, 4.4))
    plt.plot(df["noise_pct"].values, df["R2_MSFNO"].values, marker="o", linewidth=2, label="MSFNO")
    plt.plot(df["noise_pct"].values, df["R2_ResNet"].values, marker="s", linewidth=2, label="ResNet")
    plt.xlabel("RMS white noise level (%)")
    plt.ylabel("R² on validation set")
    plt.title("R² vs RMS white noise (direct prediction)")
    plt.grid(True, alpha=0.35)
    plt.ylim(0.0, 1.0)
    plt.legend()

    if save_fig:
        fig_path = os.path.join(save_dir, f"{name}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved figure to: {fig_path}")
    if show:
        plt.show()
    else:
        plt.close()

    return df


# ----------------------- Simple Execution -----------------------
if __name__ == "__main__":
    # === Config paths ===
    msfno_cfg  = "msfno-beamdi_num_t8000-run01-250814-165002.yaml"
    resnet_cfg = "resnet-beamdi_num_t8000-run01-250814-165006.yaml"

    # === Run ===
    C5S4_noise_sensitivity_direct_plot(
        MSFNO_cfgpath   = msfno_cfg,
        ResNet_cfgpath  = resnet_cfg,
        data_name       = "beamdi_num",
        validdata       = "beamdi_num_v1000",
        dir             = "./results/postprocessed",
        loc             = "C5S4",
        type            = "fig",
        name            = "R2_vs_RMS_WhiteNoise_MSFNO_vs_ResNet",
        noise_max_pct   = 0.10,     # 0~10% RMS noise
        noise_step      = 0.005,    # 0.5% step
        batch_size      = 4,        # increase batch for speed if memory allows
        seed            = 123,
        show            = True,
        save_fig        = True,
        save_csv        = True,
    )
