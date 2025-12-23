"""
Author: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
Date: 2025-08-12 18:06:19
LastEditors: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
LastEditTime: 2025-10-21 15:55:16
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


def _load_cfg(cfg_path: str) -> dict:
    with open(f"./configs/{cfg_path}", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_model_from_cfg(cfg: dict):
    key = cfg["model"]["model"]
    if key not in MODEL_CLASSES:
        raise ValueError(f"Unknown model key in cfg: {key}")
    return MODEL_CLASSES[key](**cfg["model"]["para"])


def _load_first_pt(cfg: dict) -> str:
    model_dir = os.path.join(cfg["paths"]["results_path"], "model")
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model dir not found: {model_dir}")
    pt_files = [fn for fn in os.listdir(model_dir) if fn.endswith(".pt")]
    if not pt_files:
        raise FileNotFoundError(f"No .pt found in: {model_dir}")
    return os.path.join(model_dir, pt_files[0])


def C5S3_fig01(
    interp_cfg_path,
    msfno_cfg_path,
    resnet_cfg_path,
    predictor="resnet",
    data_name="beamdi_exp",
    valid_set="beamdi_exp_scut",
    baseline="rein",
    output_dir="./results/postprocessed",
    output_type="fig",
    output_loc="C5S3",
    output_name=None,
):
    """
    Draw C5S3 figure: stiffness change (%) = (valid - baseline) for prediction and GT.
    """
    interp_cfg = _load_cfg(interp_cfg_path)
    msfno_cfg = _load_cfg(msfno_cfg_path)
    resnet_cfg = _load_cfg(resnet_cfg_path)

    predictor_key = predictor.lower()
    if predictor_key not in {"resnet", "msfno", "mosrnet"}:
        raise ValueError("predictor must be one of {'resnet','msfno','mosrnet'}")

    if predictor_key == "resnet":
        predictor_cfg = resnet_cfg
    elif predictor_key == "msfno":
        predictor_cfg = msfno_cfg
    else:
        predictor_cfg = msfno_cfg if msfno_cfg["model"]["model"] == "mosrnet" else resnet_cfg

    interp_model = _build_model_from_cfg(interp_cfg).to(DEVICE)
    interp_state = torch.load(_load_first_pt(interp_cfg), map_location=DEVICE, weights_only=True)
    interp_model.load_state_dict(interp_state)
    interp_model.eval()

    predictor_model = _build_model_from_cfg(predictor_cfg).to(DEVICE)
    pred_state = torch.load(_load_first_pt(predictor_cfg), map_location=DEVICE, weights_only=True)
    predictor_model.load_state_dict(pred_state)
    predictor_model.eval()

    valid_path = f"./datasets/{data_name}/{valid_set}.pt"
    base_path = f"./datasets/{data_name}/{baseline}.pt"
    valid_dict = torch.load(valid_path, map_location=DEVICE)
    base_dict = torch.load(base_path, map_location=DEVICE)

    in_ch = predictor_cfg["train"]["in_channels"]
    valid_mode = valid_dict["mode"][:, : max(4, in_ch), :]
    base_mode = base_dict["mode"][:, : max(4, in_ch), :]

    valid_dmg = valid_dict["dmg"].float()
    base_dmg = base_dict["dmg"].float()

    with torch.no_grad():
        out_valid = interp_model(valid_mode.to(DEVICE))
        out_base = interp_model(base_mode.to(DEVICE))

        length = out_valid.shape[-1]
        grid = torch.linspace(0.0, 1.0, steps=length, device=DEVICE).reshape(1, 1, -1)
        valid_merged = torch.cat([out_valid, grid.expand(out_valid.size(0), 1, length)], dim=1)
        base_merged = torch.cat([out_base, grid.expand(out_base.size(0), 1, length)], dim=1)

        pred_valid = predictor_model(valid_merged).mean(dim=0, keepdim=True).squeeze().detach().cpu().numpy()
        pred_base = predictor_model(base_merged).mean(dim=0, keepdim=True).squeeze().detach().cpu().numpy()

        pred_valid = 100 - pred_valid * 100
        pred_base = 100 - pred_base * 100

        gt_valid = 100 - valid_dmg[0].to(DEVICE).squeeze().detach().cpu().numpy() * 100
        gt_base = 100 - base_dmg[0].to(DEVICE).squeeze().detach().cpu().numpy() * 100

        output = np.array(pred_valid - pred_base)
        gt = np.array(gt_valid - gt_base)

    x_indices = np.linspace(0, 5400, num=output.shape[-1])
    gt_indices = np.linspace(0, 5400, num=gt.shape[-1])

    pos_mask = output > 0
    neg_mask = ~pos_mask

    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.plot(x_indices, output, color="#3A3A3A", lw=1, alpha=1.0, zorder=3, label="_nolegend_")
    ax.fill_between(
        x_indices, 0, output, where=pos_mask, facecolor="#EE7E77", alpha=1.0, zorder=2, label="_nolegend_"
    )
    ax.fill_between(
        x_indices, 0, output, where=neg_mask, facecolor="#68A7BE", alpha=1.0, zorder=2, label="_nolegend_"
    )

    ax.fill_between(gt_indices, 0, gt, facecolor="#CFCFCF", alpha=1.0, zorder=1, label="_nolegend_")
    ax.step(
        gt_indices,
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
    ax.set_ylim(-65, 165)
    ax.set_yticks(np.linspace(-50, 150, num=9))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _pos: f"{int(abs(y))}"))

    ax.axhspan(0, 165, color="#feece7", alpha=1.0, zorder=0)
    ax.axhspan(-65, 0, color="#deeeed", alpha=1.0, zorder=0)
    ax.text(
        150,
        145,
        "Stiffness Loss",
        color="#EE7E77",
        va="center",
        ha="left",
        fontsize=28,
        fontweight="bold",
    )
    ax.text(
        150,
        -45,
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
        labels=[f"{model_label}'s Prediction", "Theoretical Damage"],
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc="best",
    )

    ax.grid(True, linestyle="-", alpha=0.3, zorder=0)
    plt.tight_layout()

    save_path = os.path.join(output_dir, output_loc, output_type)
    os.makedirs(save_path, exist_ok=True)

    if output_name is None:
        output_name = f"{predictor.upper()}-Prediction_vs_GT"

    out_path = os.path.join(save_path, f"{output_name}_{valid_set}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return out_path


DEFAULT_INTERP_CFG = "mosrnet-beamdi_num_t8000-run01-250814-164957.yaml"
DEFAULT_MSFNO_CFG = "msfno-beamdi_num_t8000-run01-250814-165002.yaml"
DEFAULT_RESNET_CFG = "resnet-beamdi_num_t8000-run01-250814-165006.yaml"
DEFAULT_DATA_NAME = "beamdi_exp"
DEFAULT_BASELINE = "beamdi_exp_rein"
DEFAULT_VALID_LIST = ["beamdi_exp_mcut", "beamdi_exp_wcut", "beamdi_exp_wedg"]
DEFAULT_OUTPUT_DIR = "./results/postprocessed"
DEFAULT_OUTPUT_TYPE = "fig"
DEFAULT_OUTPUT_LOC = "C5S3"


def main():
    for valid_set in DEFAULT_VALID_LIST:
        out_path = C5S3_fig01(
            interp_cfg_path=DEFAULT_INTERP_CFG,
            msfno_cfg_path=DEFAULT_MSFNO_CFG,
            resnet_cfg_path=DEFAULT_RESNET_CFG,
            predictor="msfno",
            data_name=DEFAULT_DATA_NAME,
            valid_set=valid_set,
            baseline=DEFAULT_BASELINE,
            output_dir=DEFAULT_OUTPUT_DIR,
            output_type=DEFAULT_OUTPUT_TYPE,
            output_loc=DEFAULT_OUTPUT_LOC,
            output_name="Experimental validation results MS-FNO",
        )
        print(f"[MSFNO] Saved: {out_path}")

    for valid_set in DEFAULT_VALID_LIST:
        out_path = C5S3_fig01(
            interp_cfg_path=DEFAULT_INTERP_CFG,
            msfno_cfg_path=DEFAULT_MSFNO_CFG,
            resnet_cfg_path=DEFAULT_RESNET_CFG,
            predictor="resnet",
            data_name=DEFAULT_DATA_NAME,
            valid_set=valid_set,
            baseline=DEFAULT_BASELINE,
            output_dir=DEFAULT_OUTPUT_DIR,
            output_type=DEFAULT_OUTPUT_TYPE,
            output_loc=DEFAULT_OUTPUT_LOC,
            output_name="Experimental validation results ResNet",
        )
        print(f"[RESNET] Saved: {out_path}")


if __name__ == "__main__":
    main()
