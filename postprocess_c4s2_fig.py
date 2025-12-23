"""
Author: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
Date: 2025-08-15 19:38:31
LastEditors: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
LastEditTime: 2025-10-21 15:41:38
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
from torch.utils.data import DataLoader, TensorDataset

from models.mosrnet import MoSRNet
from models.msfno import MSFNO
from models.resnet import ResNet
from utilities.euler_bernoulli_beam_fem import BeamAnalysis


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
    key = cfg["model"]["model"].lower()
    if key not in MODEL_CLASSES:
        raise ValueError(f"Unknown model key: {key}")
    return MODEL_CLASSES[key](**cfg["model"]["para"])


def _fmt_abs_tick(y, _pos):
    """Formatter to show absolute value on y-axis."""
    return f"{int(abs(y))}"


def _load_first_pt(cfg: dict) -> str:
    model_dir = os.path.join(cfg["paths"]["results_path"], "model")
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model dir not found: {model_dir}")
    pt_files = [fn for fn in os.listdir(model_dir) if fn.endswith(".pt")]
    if not pt_files:
        raise FileNotFoundError(f"No .pt found in: {model_dir}")
    return os.path.join(model_dir, pt_files[0])


def _save_fig(fig, output_dir, output_type, output_loc, output_name):
    save_dir = os.path.join(output_dir, output_loc, output_type)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{output_name}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def C4S2_MS_FNO_signed_error_plus_intact(
    interp_cfg_path,
    msfno_cfg_path,
    resnet_cfg_path,
    data_name="beamdi_num",
    valid_set="beamdi_num_v1000",
    down_idx=None,
    batch_size=32,
    L_beam=5.4,
    E=210e9,
    I=57.48e-8,
    rho=7850.0,
    A=65.423 * 0.0001,
    n_elem=540,
    output_dir="./results/postprocessed",
    output_type="fig",
    output_loc="C4S2",
    output_name="C4S2-MS-FNO_SignedError+Intact",
):
    """
    Plot signed mean error + intact prediction using MSFNO.
    """
    # --- load configs & models ---
    interp_cfg = _load_cfg(interp_cfg_path)
    msfno_cfg = _load_cfg(msfno_cfg_path)
    resnet_cfg = _load_cfg(resnet_cfg_path)

    interp_model = _build_model_from_cfg(interp_cfg).to(DEVICE)
    interp_model.load_state_dict(
        torch.load(_load_first_pt(interp_cfg), map_location=DEVICE, weights_only=True)
    )
    interp_model.eval()

    msfno_model = _build_model_from_cfg(msfno_cfg).to(DEVICE)
    msfno_model.load_state_dict(
        torch.load(_load_first_pt(msfno_cfg), map_location=DEVICE, weights_only=True)
    )
    msfno_model.eval()

    # --- dataset ---
    ds_path = f"./datasets/{data_name}/{valid_set}.pt"
    ds = torch.load(ds_path, map_location=DEVICE)
    in_ch = max(4, msfno_cfg["train"]["in_channels"], resnet_cfg["train"]["in_channels"])
    modes = ds["mode"][:, :in_ch, :]
    dmg = ds["dmg"].float()

    if down_idx is None:
        down_idx = np.linspace(0, modes.shape[-1] - 1, num=9, dtype=int)
    down_idx = np.asarray(down_idx, dtype=int)

    # --- signed mean error (validation) ---
    X_down = modes[:, :, down_idx]
    Y_full = dmg
    loader = DataLoader(TensorDataset(X_down.float(), Y_full.float()), batch_size=batch_size, shuffle=False)

    err_sum = None
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            x_rec = interp_model(xb)
            batch_size_local, _, rec_len = x_rec.shape
            grid = (
                torch.linspace(0.0, 1.0, steps=rec_len, device=DEVICE)
                .reshape(1, 1, -1)
                .expand(batch_size_local, 1, rec_len)
            )
            x_in = torch.cat([x_rec, grid], dim=1)

            if yb.shape[-1] != rec_len:
                y_aligned = torch.nn.functional.interpolate(
                    yb, size=rec_len, mode="linear", align_corners=False
                )
            else:
                y_aligned = yb

            gt_pct = (100 - y_aligned.squeeze(1) * 100).detach().cpu().numpy()
            pred_ms = (100 - msfno_model(x_in).squeeze(1) * 100).detach().cpu().numpy()

            err = pred_ms.squeeze() - gt_pct
            batch_sum = err.sum(axis=0)

            if err_sum is None:
                err_sum = batch_sum
            else:
                if err_sum.shape[0] != batch_sum.shape[0]:
                    x_old = np.linspace(0, 1, num=batch_sum.shape[0])
                    x_new = np.linspace(0, 1, num=err_sum.shape[0])
                    batch_sum = np.interp(x_new, x_old, batch_sum)
                err_sum += batch_sum

            total += batch_size_local

    signed_mean_err = err_sum / total
    x_mm_err = np.linspace(0, 5400, num=signed_mean_err.shape[0])

    # --- intact FEM -> downsample -> reconstruct -> predictor ---
    intact = np.ones(n_elem)
    beam = BeamAnalysis(L_beam, E, I, rho, A, n_elem)
    beam.assemble_matrices(dmgfield=intact, mass_dmg_power=0)
    beam.apply_BC()
    _, eigenvectors = beam.solve_eigenproblem()
    u_vectors, _ = beam.split_eigenvectors(eigenvectors)
    modes3 = u_vectors[:, 2:5].T

    for i in range(modes3.shape[0]):
        if modes3[i, 14] < 0:
            modes3[i] = -modes3[i]
        max_abs = np.max(np.abs(modes3[i]))
        if max_abs > 0:
            modes3[i] = modes3[i] / max_abs

    modes_down = torch.tensor(modes3[:, down_idx], dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        modes_rec = interp_model(modes_down)
    modes_rec = modes_rec.squeeze(0)
    rec_len = modes_rec.shape[-1]
    grid = torch.linspace(0.0, 1.0, steps=rec_len, device=DEVICE).reshape(1, 1, -1).squeeze(0)
    x_in_int = torch.cat([modes_rec, grid], dim=0).unsqueeze(0)

    with torch.no_grad():
        pred_int_m = (100 - msfno_model(x_in_int).squeeze() * 100).detach().cpu().numpy()

    x_mm_int = np.linspace(0, 5400, num=pred_int_m.shape[-1])

    # --- plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        x_mm_err,
        signed_mean_err,
        lw=2,
        color="#d62728",
        linestyle="-",
        label="Mean error on the validation set (MoSRNet + MS-FNO)",
    )
    ax.plot(
        x_mm_int,
        pred_int_m,
        lw=2,
        color="#1f77b4",
        linestyle="--",
        label="Prediction of intact scenario (MoSRNet + MS-FNO)",
    )

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

    return _save_fig(fig, output_dir, output_type, output_loc, output_name)


def C4S2_ResNet_signed_error_plus_intact(
    interp_cfg_path,
    msfno_cfg_path,
    resnet_cfg_path,
    data_name="beamdi_num",
    valid_set="beamdi_num_v1000",
    down_idx=None,
    batch_size=32,
    L_beam=5.4,
    E=210e9,
    I=57.48e-8,
    rho=7850.0,
    A=65.423 * 0.0001,
    n_elem=540,
    output_dir="./results/postprocessed",
    output_type="fig",
    output_loc="C4S2",
    output_name="C4S2-ResNet_SignedError+Intact",
):
    """
    Plot signed mean error + intact prediction using ResNet.
    """
    # --- load configs & models ---
    interp_cfg = _load_cfg(interp_cfg_path)
    msfno_cfg = _load_cfg(msfno_cfg_path)
    resnet_cfg = _load_cfg(resnet_cfg_path)

    interp_model = _build_model_from_cfg(interp_cfg).to(DEVICE)
    interp_model.load_state_dict(
        torch.load(_load_first_pt(interp_cfg), map_location=DEVICE, weights_only=True)
    )
    interp_model.eval()

    resnet_model = _build_model_from_cfg(resnet_cfg).to(DEVICE)
    resnet_model.load_state_dict(
        torch.load(_load_first_pt(resnet_cfg), map_location=DEVICE, weights_only=True)
    )
    resnet_model.eval()

    # --- dataset ---
    ds_path = f"./datasets/{data_name}/{valid_set}.pt"
    ds = torch.load(ds_path, map_location=DEVICE)
    in_ch = max(4, msfno_cfg["train"]["in_channels"], resnet_cfg["train"]["in_channels"])
    modes = ds["mode"][:, :in_ch, :]
    dmg = ds["dmg"].float()

    if down_idx is None:
        down_idx = np.linspace(0, modes.shape[-1] - 1, num=9, dtype=int)
    down_idx = np.asarray(down_idx, dtype=int)

    # --- signed mean error (validation) ---
    X_down = modes[:, :, down_idx]
    Y_full = dmg
    loader = DataLoader(TensorDataset(X_down.float(), Y_full.float()), batch_size=batch_size, shuffle=False)

    err_sum = None
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            x_rec = interp_model(xb)
            batch_size_local, _, rec_len = x_rec.shape
            grid = (
                torch.linspace(0.0, 1.0, steps=rec_len, device=DEVICE)
                .reshape(1, 1, -1)
                .expand(batch_size_local, 1, rec_len)
            )
            x_in = torch.cat([x_rec, grid], dim=1)

            if yb.shape[-1] != rec_len:
                y_aligned = torch.nn.functional.interpolate(
                    yb, size=rec_len, mode="linear", align_corners=False
                )
            else:
                y_aligned = yb

            gt_pct = (100 - y_aligned.squeeze(1) * 100).detach().cpu().numpy()
            pred_rs = (100 - resnet_model(x_in).squeeze(1) * 100).detach().cpu().numpy()

            err = pred_rs.squeeze() - gt_pct
            batch_sum = err.sum(axis=0)

            if err_sum is None:
                err_sum = batch_sum
            else:
                if err_sum.shape[0] != batch_sum.shape[0]:
                    x_old = np.linspace(0, 1, num=batch_sum.shape[0])
                    x_new = np.linspace(0, 1, num=err_sum.shape[0])
                    batch_sum = np.interp(x_new, x_old, batch_sum)
                err_sum += batch_sum

            total += batch_size_local

    signed_mean_err = err_sum / total
    x_mm_err = np.linspace(0, 5400, num=signed_mean_err.shape[0])

    # --- intact FEM -> downsample -> reconstruct -> predictor ---
    intact = np.ones(n_elem)
    beam = BeamAnalysis(L_beam, E, I, rho, A, n_elem)
    beam.assemble_matrices(dmgfield=intact, mass_dmg_power=0)
    beam.apply_BC()
    _, eigenvectors = beam.solve_eigenproblem()
    u_vectors, _ = beam.split_eigenvectors(eigenvectors)
    modes3 = u_vectors[:, 2:5].T

    for i in range(modes3.shape[0]):
        if modes3[i, 14] < 0:
            modes3[i] = -modes3[i]
        max_abs = np.max(np.abs(modes3[i]))
        if max_abs > 0:
            modes3[i] = modes3[i] / max_abs

    modes_down = torch.tensor(modes3[:, down_idx], dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        modes_rec = interp_model(modes_down)
    modes_rec = modes_rec.squeeze(0)
    rec_len = modes_rec.shape[-1]
    grid = torch.linspace(0.0, 1.0, steps=rec_len, device=DEVICE).reshape(1, 1, -1).squeeze(0)
    x_in_int = torch.cat([modes_rec, grid], dim=0).unsqueeze(0)

    with torch.no_grad():
        pred_int_r = (100 - resnet_model(x_in_int).squeeze() * 100).detach().cpu().numpy()

    x_mm_int = np.linspace(0, 5400, num=pred_int_r.shape[-1])

    # --- plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        x_mm_err,
        signed_mean_err,
        lw=2,
        color="#d62728",
        linestyle="-",
        label="Mean error on the validation set (MoSRNet + ResNet)",
    )
    ax.plot(
        x_mm_int,
        pred_int_r,
        lw=2,
        color="#1f77b4",
        linestyle="--",
        label="Prediction of intact scenario (MoSRNet + ResNet)",
    )

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

    return _save_fig(fig, output_dir, output_type, output_loc, output_name)


def C4S2_reconstructed_withbaseline(
    interp_cfg_path,
    msfno_cfg_path,
    resnet_cfg_path,
    predictor="resnet",
    data_name="beamdi_num",
    subset_name="beamdi_num_v1000",
    sample_index=34,
    down_idx=None,
    output_dir="./results/postprocessed",
    L_beam=5.4,
    E=210e9,
    I=57.48e-8,
    rho=7850.0,
    A=65.423 * 0.0001,
    n_elem=540,
    output_type="fig",
    output_loc="C4S2",
    output_name="Reconstructed-input prediction vs Ground Truth after bias compensation",
):
    """
    Plot reconstructed prediction vs ground truth after baseline compensation.
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
    interp_state = torch.load(_load_first_pt(interp_cfg), map_location=DEVICE, weights_only=True)
    interp_model.load_state_dict(interp_state)
    interp_model.eval()

    predictor_model = _build_model_from_cfg(predictor_cfg).to(DEVICE)
    pred_state = torch.load(_load_first_pt(predictor_cfg), map_location=DEVICE, weights_only=True)
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
    if down_idx is None:
        down_idx = np.linspace(0, length - 1, num=9, dtype=int)

    # --- reconstruct sample ---
    with torch.no_grad():
        x_hr = modes[sample_index : sample_index + 1]
        x_ds = x_hr[:, :, down_idx]
        x_rec = interp_model(x_ds.to(DEVICE))

        rec_len = x_rec.shape[-1]
        grid = torch.linspace(0.0, 1.0, steps=rec_len, device=DEVICE).reshape(1, 1, -1)
        x_in = torch.cat([x_rec, grid], dim=1)

        pred = predictor_model(x_in)
        pred = 100 - pred.squeeze().detach().cpu().numpy() * 100
        gt = 100 - dmg[sample_index].squeeze().detach().cpu().numpy() * 100

    # --- intact FEM baseline ---
    intact = np.ones(n_elem)
    beam = BeamAnalysis(L_beam, E, I, rho, A, n_elem)
    beam.assemble_matrices(dmgfield=intact, mass_dmg_power=0)
    beam.apply_BC()
    _, eigenvectors = beam.solve_eigenproblem()
    u_vectors, _ = beam.split_eigenvectors(eigenvectors)
    modes3 = u_vectors[:, 2:5].T

    for i in range(modes3.shape[0]):
        if modes3[i, 14] < 0:
            modes3[i] = -modes3[i]
        max_abs = np.max(np.abs(modes3[i]))
        if max_abs > 0:
            modes3[i] = modes3[i] / max_abs

    modes_down = torch.tensor(modes3[:, down_idx], dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        modes_rec = interp_model(modes_down)
    modes_rec = modes_rec.squeeze(0)
    rec_len = modes_rec.shape[-1]
    grid = torch.linspace(0.0, 1.0, steps=rec_len, device=DEVICE).reshape(1, 1, -1).squeeze(0)
    x_in_int = torch.cat([modes_rec, grid], dim=0).unsqueeze(0)

    with torch.no_grad():
        pred_int = (100 - predictor_model(x_in_int).squeeze() * 100).detach().cpu().numpy()

    # --- plot ---
    x_idx = np.linspace(0, 5400, num=pred.shape[-1])
    gt_idx = np.linspace(0, 5400, num=gt.shape[-1])

    pred = pred - pred_int
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
        output_name = f"C5S1_Reconstructed_{pred_key.upper()}"
    name_full = f"{output_name}-{subset_name}-idx{sample_index}"
    return _save_fig(fig, output_dir, output_type, output_loc, name_full)


DEFAULT_INTERP_CFG = "mosrnet-beamdi_num_t8000-run01-250814-164957.yaml"
DEFAULT_MSFNO_CFG = "msfno-beamdi_num_t8000-run01-250814-165002.yaml"
DEFAULT_RESNET_CFG = "resnet-beamdi_num_t8000-run01-250814-165006.yaml"
DEFAULT_SAMPLE_INDEX = 34
DEFAULT_DATA_NAME = "beamdi_num"
DEFAULT_VALID_SET = "beamdi_num_v1000"
DEFAULT_DOWN_IDX = np.array([0, 68, 135, 203, 270, 337, 405, 473, 540], dtype=int)
DEFAULT_OUTPUT_DIR = "./results/postprocessed"
DEFAULT_OUTPUT_TYPE = "fig"
DEFAULT_OUTPUT_LOC = "C4S2"


def main():
    out1 = C4S2_MS_FNO_signed_error_plus_intact(
        interp_cfg_path=DEFAULT_INTERP_CFG,
        msfno_cfg_path=DEFAULT_MSFNO_CFG,
        resnet_cfg_path=DEFAULT_RESNET_CFG,
        data_name=DEFAULT_DATA_NAME,
        valid_set=DEFAULT_VALID_SET,
        down_idx=DEFAULT_DOWN_IDX,
        batch_size=32,
        output_dir=DEFAULT_OUTPUT_DIR,
        output_type=DEFAULT_OUTPUT_TYPE,
        output_loc=DEFAULT_OUTPUT_LOC,
        output_name="C4S2-MS-FNO_SignedError+Intact",
    )
    print("[C4S2] Saved:", out1)

    out2 = C4S2_ResNet_signed_error_plus_intact(
        interp_cfg_path=DEFAULT_INTERP_CFG,
        msfno_cfg_path=DEFAULT_MSFNO_CFG,
        resnet_cfg_path=DEFAULT_RESNET_CFG,
        data_name=DEFAULT_DATA_NAME,
        valid_set=DEFAULT_VALID_SET,
        down_idx=DEFAULT_DOWN_IDX,
        batch_size=32,
        output_dir=DEFAULT_OUTPUT_DIR,
        output_type=DEFAULT_OUTPUT_TYPE,
        output_loc=DEFAULT_OUTPUT_LOC,
        output_name="C4S2-ResNet_SignedError+Intact",
    )
    print("[C4S2] Saved:", out2)

    p3 = C4S2_reconstructed_withbaseline(
        interp_cfg_path=DEFAULT_INTERP_CFG,
        msfno_cfg_path=DEFAULT_MSFNO_CFG,
        resnet_cfg_path=DEFAULT_RESNET_CFG,
        predictor="msfno",
        data_name=DEFAULT_DATA_NAME,
        subset_name=DEFAULT_VALID_SET,
        sample_index=DEFAULT_SAMPLE_INDEX,
        down_idx=None,
        output_dir=DEFAULT_OUTPUT_DIR,
        L_beam=5.4,
        E=210e9,
        I=57.48e-8,
        rho=7850.0,
        A=65.423 * 0.0001,
        n_elem=540,
        output_type="fig",
        output_loc="C4S2",
        output_name="numerical validation results after bias compensation_Itp_MSFNO",
    )
    print("Saved:", p3)

    p4 = C4S2_reconstructed_withbaseline(
        interp_cfg_path=DEFAULT_INTERP_CFG,
        msfno_cfg_path=DEFAULT_MSFNO_CFG,
        resnet_cfg_path=DEFAULT_RESNET_CFG,
        predictor="resnet",
        data_name=DEFAULT_DATA_NAME,
        subset_name=DEFAULT_VALID_SET,
        sample_index=DEFAULT_SAMPLE_INDEX,
        down_idx=None,
        output_dir=DEFAULT_OUTPUT_DIR,
        L_beam=5.4,
        E=210e9,
        I=57.48e-8,
        rho=7850.0,
        A=65.423 * 0.0001,
        n_elem=540,
        output_type="fig",
        output_loc="C4S2",
        output_name="numerical validation results after bias compensation_Itp_RESNET",
    )
    print("Saved:", p4)


if __name__ == "__main__":
    main()
