"""
Author: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
Date: 2025-08-15 18:32:14
LastEditors: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
LastEditTime: 2025-10-21 15:34:08
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch as MplPatch
from matplotlib.ticker import FuncFormatter

from models.mosrnet import MoSRNet
from models.msfno import MSFNO
from models.resnet import ResNet


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


MODEL_CLASSES = {"msfno": MSFNO, "resnet": ResNet, "mosrnet": MoSRNet}
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _build_model_from_cfg(cfg: dict):
    """Instantiate a model from cfg['model']."""
    key = cfg["model"]["model"]
    if key not in MODEL_CLASSES:
        raise ValueError(f"Unknown model key: {key}")
    return MODEL_CLASSES[key](**cfg["model"]["para"])


def _load_first_pt(cfg: dict) -> str:
    """Return the first .pt file under {results_path}/model/."""
    model_dir = os.path.join(cfg["paths"]["results_path"], "model")
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model dir not found: {model_dir}")
    pt_files = [fn for fn in os.listdir(model_dir) if fn.endswith(".pt")]
    if not pt_files:
        raise FileNotFoundError(f"No .pt weights in: {model_dir}")
    return os.path.join(model_dir, pt_files[0])


def _load_cfg(cfg_path: str) -> dict:
    with open(f"./configs/{cfg_path}", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _fmt_abs_tick(y, _pos):
    """Formatter to show absolute value on y-axis."""
    return f"{int(abs(y))}"


def _save_figure(fig, output_dir, output_type, output_loc, output_name):
    """Save figure to {output_dir}/{output_loc}/{output_type}/{output_name}.png."""
    save_dir = os.path.join(output_dir, output_loc, output_type)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{output_name}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def C4S1_fig01_original(
    interp_cfg_path,
    msfno_cfg_path,
    resnet_cfg_path,
    predictor="resnet",
    data_name="beamdi_num",
    subset_name="beamdi_num_v1000",
    sample_index=34,
    output_dir="./results/postprocessed",
    output_type="fig",
    output_loc="C4S1",
    output_name="Original-input prediction vs Ground Truth",
):
    """
    Plot C4S1 (original data): predictor consumes high-res modes + grid channel.
    No baseline; model output is directly used. Ground truth is drawn as band + step curve.
    """
    # --- load configs ---
    interp_cfg = _load_cfg(interp_cfg_path)
    msfno_cfg = _load_cfg(msfno_cfg_path)
    resnet_cfg = _load_cfg(resnet_cfg_path)

    pred_key = predictor.lower()
    if pred_key not in {"resnet", "msfno", "mosrnet"}:
        raise ValueError("predictor must be in {'resnet','msfno','mosrnet'}")
    if pred_key == "resnet":
        predictor_cfg = resnet_cfg
    elif pred_key == "msfno":
        predictor_cfg = msfno_cfg
    else:
        predictor_cfg = msfno_cfg if msfno_cfg["model"]["model"] == "mosrnet" else resnet_cfg

    # --- build & load predictor ---
    predictor_model = _build_model_from_cfg(predictor_cfg).to(DEVICE)
    pred_pt = _load_first_pt(predictor_cfg)
    pred_state = torch.load(pred_pt, map_location=DEVICE, weights_only=True)
    predictor_model.load_state_dict(pred_state)
    predictor_model.eval()

    # --- load dataset ---
    ds_path = f"./datasets/{data_name}/{subset_name}.pt"
    ds = torch.load(ds_path, map_location=DEVICE)
    in_ch = predictor_cfg["train"]["in_channels"]
    modes = ds["mode"][:, :in_ch, :]
    dmg = ds["dmg"].float()

    # --- prepare one sample ---
    with torch.no_grad():
        x_hr = modes[sample_index : sample_index + 1]  # [1, C, L]
        length = x_hr.shape[-1]
        grid = torch.linspace(0.0, 1.0, steps=length, device=DEVICE).reshape(1, 1, -1)
        x_in = torch.cat([x_hr.to(DEVICE), grid], dim=1)

        pred = predictor_model(x_in)
        pred = 100 - pred.squeeze().detach().cpu().numpy() * 100
        gt = 100 - dmg[sample_index].squeeze().detach().cpu().numpy() * 100

    # --- plot ---
    x_idx = np.linspace(0, 5400, num=pred.shape[-1])
    gt_idx = np.linspace(0, 5400, num=gt.shape[-1])

    pos_mask = pred > 0
    neg_mask = ~pos_mask

    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.plot(x_idx, pred, color="#3A3A3A", lw=1, alpha=1.0, zorder=3, label="_nolegend_")
    ax.fill_between(
        x_idx, 0, pred, where=pos_mask, facecolor="#EE7E77", alpha=1.0, zorder=2, label="_nolegend_"
    )
    ax.fill_between(
        x_idx, 0, pred, where=neg_mask, facecolor="#68A7BE", alpha=1.0, zorder=2, label="_nolegend_"
    )

    ax.fill_between(gt_idx, 0, gt, facecolor="#CFCFCF", alpha=1.0, zorder=1, label="_nolegend_")
    ax.step(
        gt_idx,
        gt,
        color="#3A3A3A",
        lw=1,
        linestyle="--",
        alpha=1.0,
        where="mid",
        zorder=4,
        label="_nolegend_",
    )

    ax.axhline(0, color="k", linewidth=1.5, zorder=10)
    ax.set_xlim(0, 5400)
    ax.set_xticks(np.linspace(0, 5400, num=9))
    ax.set_ylim(-85, 85)
    ax.set_yticks(np.linspace(-75, 75, num=7))
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_abs_tick))

    ax.axhspan(0, 85, color="#feece7", alpha=1.0, zorder=0)
    ax.axhspan(-85, 0, color="#deeeed", alpha=1.0, zorder=0)

    ax.text(150, 65, "Stiffness Loss", color="#EE7E77", va="center", ha="left", fontsize=28, fontweight="bold")
    ax.text(
        150,
        -65,
        "Stiffness Increase",
        color="#68A7BE",
        va="center",
        ha="left",
        fontsize=28,
        fontweight="bold",
    )

    ax.set_ylabel("Stiffness Change (%)")
    ax.set_xlabel("Beam Span (mm)")

    red_patch = MplPatch(color="#EE7E77")
    blue_patch = MplPatch(color="#68A7BE")
    gt_patch = MplPatch(facecolor="#CFCFCF")

    if predictor == "msfno":
        model_label = "MS-FNO"
    elif predictor == "resnet":
        model_label = "ResNet"
    else:
        raise KeyError(f"no model named {predictor}")

    ax.legend(
        handles=[(red_patch, blue_patch), gt_patch],
        labels=[f"{model_label}'s Prediction", "Ground Truth"],
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc="upper right",
    )

    ax.grid(True, linestyle="-", alpha=0.3, zorder=0)
    plt.tight_layout()

    if output_name is None:
        output_name = f"C4S1_Original_{pred_key.upper()}"
    name_full = f"{output_name}-{subset_name}-idx{sample_index}"
    return _save_figure(fig, output_dir, output_type, output_loc, name_full)


def C4S1_fig02_reconstructed(
    interp_cfg_path,
    msfno_cfg_path,
    resnet_cfg_path,
    predictor="resnet",
    data_name="beamdi_num",
    subset_name="beamdi_num_v1000",
    sample_index=34,
    down_indices=None,
    output_dir="./results/postprocessed",
    output_type="fig",
    output_loc="C4S1",
    output_name="Reconstructed-input prediction vs Ground Truth",
):
    """
    Plot C4S1 (reconstructed): downsample -> reconstruction -> predictor.
    No baseline; model output is directly used. Ground truth is drawn as band + step curve.
    """
    # --- load configs ---
    interp_cfg = _load_cfg(interp_cfg_path)
    msfno_cfg = _load_cfg(msfno_cfg_path)
    resnet_cfg = _load_cfg(resnet_cfg_path)

    pred_key = predictor.lower()
    if pred_key not in {"resnet", "msfno", "mosrnet"}:
        raise ValueError("predictor must be in {'resnet','msfno','mosrnet'}")
    if pred_key == "resnet":
        predictor_cfg = resnet_cfg
    elif pred_key == "msfno":
        predictor_cfg = msfno_cfg
    else:
        predictor_cfg = msfno_cfg if msfno_cfg["model"]["model"] == "mosrnet" else resnet_cfg

    # --- build & load models ---
    interp_model = _build_model_from_cfg(interp_cfg).to(DEVICE)
    interp_pt = _load_first_pt(interp_cfg)
    interp_state = torch.load(interp_pt, map_location=DEVICE, weights_only=True)
    interp_model.load_state_dict(interp_state)
    interp_model.eval()

    predictor_model = _build_model_from_cfg(predictor_cfg).to(DEVICE)
    pred_pt = _load_first_pt(predictor_cfg)
    pred_state = torch.load(pred_pt, map_location=DEVICE, weights_only=True)
    predictor_model.load_state_dict(pred_state)
    predictor_model.eval()

    # --- load dataset ---
    ds_path = f"./datasets/{data_name}/{subset_name}.pt"
    ds = torch.load(ds_path, map_location=DEVICE)
    in_ch = predictor_cfg["train"]["in_channels"]
    modes = ds["mode"][:, :in_ch, :]
    dmg = ds["dmg"].float()

    # --- define downsample indices ---
    length = modes.shape[-1]
    if down_indices is None:
        down_indices = np.linspace(0, length - 1, num=9, dtype=int)

    # --- reconstruct sample ---
    with torch.no_grad():
        x_hr = modes[sample_index : sample_index + 1]
        x_ds = x_hr[:, :, down_indices]
        x_rec = interp_model(x_ds.to(DEVICE))

        length_rec = x_rec.shape[-1]
        grid = torch.linspace(0.0, 1.0, steps=length_rec, device=DEVICE).reshape(1, 1, -1)
        x_in = torch.cat([x_rec, grid], dim=1)

        pred = predictor_model(x_in)
        pred = 100 - pred.squeeze().detach().cpu().numpy() * 100
        gt = 100 - dmg[sample_index].squeeze().detach().cpu().numpy() * 100

    # --- plot ---
    x_idx = np.linspace(0, 5400, num=pred.shape[-1])
    gt_idx = np.linspace(0, 5400, num=gt.shape[-1])

    pos_mask = pred > 0
    neg_mask = ~pos_mask

    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.plot(x_idx, pred, color="#3A3A3A", lw=1, alpha=1.0, zorder=3, label="_nolegend_")
    ax.fill_between(
        x_idx, 0, pred, where=pos_mask, facecolor="#EE7E77", alpha=1.0, zorder=2, label="_nolegend_"
    )
    ax.fill_between(
        x_idx, 0, pred, where=neg_mask, facecolor="#68A7BE", alpha=1.0, zorder=2, label="_nolegend_"
    )

    ax.fill_between(gt_idx, 0, gt, facecolor="#CFCFCF", alpha=1.0, zorder=1, label="_nolegend_")
    ax.step(
        gt_idx,
        gt,
        color="#3A3A3A",
        lw=1,
        linestyle="--",
        alpha=1.0,
        where="mid",
        zorder=4,
        label="_nolegend_",
    )

    ax.axhline(0, color="k", linewidth=1.5, zorder=10)
    ax.set_xlim(0, 5400)
    ax.set_xticks(np.linspace(0, 5400, num=9))
    ax.set_ylim(-85, 85)
    ax.set_yticks(np.linspace(-75, 75, num=7))
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_abs_tick))

    ax.axhspan(0, 85, color="#feece7", alpha=1.0, zorder=0)
    ax.axhspan(-85, 0, color="#deeeed", alpha=1.0, zorder=0)

    ax.text(150, 65, "Stiffness Loss", color="#EE7E77", va="center", ha="left", fontsize=28, fontweight="bold")
    ax.text(
        150,
        -65,
        "Stiffness Increase",
        color="#68A7BE",
        va="center",
        ha="left",
        fontsize=28,
        fontweight="bold",
    )

    ax.set_ylabel("Stiffness Change (%)")
    ax.set_xlabel("Beam Span (mm)")

    red_patch = MplPatch(color="#EE7E77")
    blue_patch = MplPatch(color="#68A7BE")
    gt_patch = MplPatch(facecolor="#CFCFCF")

    if predictor == "msfno":
        model_label = "MS-FNO"
    elif predictor == "resnet":
        model_label = "ResNet"
    else:
        raise KeyError(f"no model named {predictor}")

    ax.legend(
        handles=[(red_patch, blue_patch), gt_patch],
        labels=[f"{model_label}'s Prediction", "Ground Truth"],
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc="upper right",
    )

    ax.grid(True, linestyle="-", alpha=0.3, zorder=0)
    plt.tight_layout()

    if output_name is None:
        output_name = f"C4S1_Reconstructed_{pred_key.upper()}"
    name_full = f"{output_name}-{subset_name}-idx{sample_index}"
    return _save_figure(fig, output_dir, output_type, output_loc, name_full)


DEFAULT_INTERP_CFG = "mosrnet-beamdi_num_t8000-run01-250814-164957.yaml"
DEFAULT_MSFNO_CFG = "msfno-beamdi_num_t8000-run01-250814-165002.yaml"
DEFAULT_RESNET_CFG = "resnet-beamdi_num_t8000-run01-250814-165006.yaml"
DEFAULT_SAMPLE_INDEX = 34


def main():
    p1 = C4S1_fig01_original(
        interp_cfg_path=DEFAULT_INTERP_CFG,
        msfno_cfg_path=DEFAULT_MSFNO_CFG,
        resnet_cfg_path=DEFAULT_RESNET_CFG,
        predictor="msfno",
        data_name="beamdi_num",
        subset_name="beamdi_num_v1000",
        sample_index=DEFAULT_SAMPLE_INDEX,
        output_dir="./results/postprocessed",
        output_type="fig",
        output_loc="C4S1",
        output_name="numerical validation results _Ori_MSFNO",
    )
    print("Saved:", p1)

    p2 = C4S1_fig01_original(
        interp_cfg_path=DEFAULT_INTERP_CFG,
        msfno_cfg_path=DEFAULT_MSFNO_CFG,
        resnet_cfg_path=DEFAULT_RESNET_CFG,
        predictor="resnet",
        data_name="beamdi_num",
        subset_name="beamdi_num_v1000",
        sample_index=DEFAULT_SAMPLE_INDEX,
        output_dir="./results/postprocessed",
        output_type="fig",
        output_loc="C4S1",
        output_name="numerical validation results _Ori_RESNET",
    )
    print("Saved:", p2)

    p3 = C4S1_fig02_reconstructed(
        interp_cfg_path=DEFAULT_INTERP_CFG,
        msfno_cfg_path=DEFAULT_MSFNO_CFG,
        resnet_cfg_path=DEFAULT_RESNET_CFG,
        predictor="msfno",
        data_name="beamdi_num",
        subset_name="beamdi_num_v1000",
        sample_index=DEFAULT_SAMPLE_INDEX,
        down_indices=None,
        output_dir="./results/postprocessed",
        output_type="fig",
        output_loc="C4S1",
        output_name="numerical validation results _Itp_MSFNO",
    )
    print("Saved:", p3)

    p4 = C4S1_fig02_reconstructed(
        interp_cfg_path=DEFAULT_INTERP_CFG,
        msfno_cfg_path=DEFAULT_MSFNO_CFG,
        resnet_cfg_path=DEFAULT_RESNET_CFG,
        predictor="resnet",
        data_name="beamdi_num",
        subset_name="beamdi_num_v1000",
        sample_index=DEFAULT_SAMPLE_INDEX,
        down_indices=None,
        output_dir="./results/postprocessed",
        output_type="fig",
        output_loc="C4S1",
        output_name="numerical validation results _Itp_RESNET",
    )
    print("Saved:", p4)


if __name__ == "__main__":
    main()
