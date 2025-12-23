"""
Author: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
Date: 2025-08-18 16:51:27
LastEditors: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
LastEditTime: 2025-10-21 15:37:19
"""

import os

import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from losses.r2 import R2
from models.mosrnet import MoSRNet
from models.msfno import MSFNO
from models.resnet import ResNet
from utilities.euler_bernoulli_beam_fem import BeamAnalysis


def set_plot_style():
    """Apply global Matplotlib style for postprocess scripts."""
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


def _load_first_pt(cfg: dict) -> str:
    model_dir = os.path.join(cfg["paths"]["results_path"], "model")
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model dir not found: {model_dir}")
    pt_files = [fn for fn in os.listdir(model_dir) if fn.endswith(".pt")]
    if not pt_files:
        raise FileNotFoundError(f"No .pt found in: {model_dir}")
    return os.path.join(model_dir, pt_files[0])


def C4S2_r2_before_biascompen(
    interp_cfg_path,
    msfno_cfg_path,
    resnet_cfg_path,
    predictor="msfno",
    data_name="beamdi_num",
    valid_set="beamdi_num_v1000",
    down_idx=None,
    output_dir="./results/postprocessed",
    output_type="csv",
    output_loc="C4S2",
    output_name="C4S2-MS-FNO_r2_before",
):
    """Compute R2 on the validation set before bias compensation."""
    interp_cfg = _load_cfg(interp_cfg_path)
    msfno_cfg = _load_cfg(msfno_cfg_path)
    resnet_cfg = _load_cfg(resnet_cfg_path)

    interp_model = _build_model_from_cfg(interp_cfg).to(DEVICE)
    interp_model.load_state_dict(
        torch.load(_load_first_pt(interp_cfg), map_location=DEVICE, weights_only=True)
    )
    interp_model.eval()

    pred_key = predictor.lower()
    if pred_key not in {"resnet", "msfno", "mosrnet"}:
        raise ValueError("predictor must be in {'resnet','msfno','mosrnet'}")
    if pred_key == "resnet":
        predictor_cfg = resnet_cfg
    elif pred_key == "msfno":
        predictor_cfg = msfno_cfg
    else:
        raise KeyError("the pred model must be msfno or resnet")

    pred_model = _build_model_from_cfg(predictor_cfg).to(DEVICE)
    pred_model.load_state_dict(
        torch.load(_load_first_pt(predictor_cfg), map_location=DEVICE, weights_only=True)
    )
    pred_model.eval()

    ds_path = f"./datasets/{data_name}/{valid_set}.pt"
    ds = torch.load(ds_path, map_location=DEVICE)
    in_ch = max(3, msfno_cfg["train"]["in_channels"], resnet_cfg["train"]["in_channels"])
    modes = ds["mode"][:, :in_ch, :]
    dmg = ds["dmg"].float()

    if down_idx is None:
        down_idx = np.linspace(0, modes.shape[-1] - 1, num=9, dtype=int)
    down_idx = np.asarray(down_idx, dtype=int)

    loader = DataLoader(TensorDataset(modes[:, :, down_idx].float(), dmg.float()), batch_size=1, shuffle=False)

    r2_metric = R2()
    results = []

    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            x_rec = interp_model(x)
            batch_size, _, rec_len = x_rec.shape
            grid = (
                torch.linspace(0.0, 1.0, steps=rec_len, device=DEVICE)
                .reshape(1, 1, -1)
                .expand(batch_size, 1, rec_len)
            )
            x_in = torch.cat([x_rec, grid], dim=1)

            if y.shape[-1] != rec_len:
                y_aligned = torch.nn.functional.interpolate(
                    y, size=rec_len, mode="linear", align_corners=False
                )
            else:
                y_aligned = y

            y_pred = pred_model(x_in)
            r2_loss = r2_metric(
                y_pred.view(y_pred.shape[0], -1), y_aligned.view(y_aligned.shape[0], -1)
            )
            results.append(r2_loss.detach().cpu().item())

    mean_r2 = np.mean(results)
    df = pd.DataFrame({"R2_each": results, "R2_mean": [mean_r2] * len(results)})

    save_dir = os.path.join(output_dir, output_loc, output_type)
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{output_name}.csv")
    df.to_csv(csv_path, index=False)

    print(f"Saved results to {csv_path}")
    return df


def C4S2_r2_after_biascompen(
    interp_cfg_path,
    msfno_cfg_path,
    resnet_cfg_path,
    predictor="msfno",
    data_name="beamdi_num",
    valid_set="beamdi_num_v1000",
    down_idx=None,
    output_dir="./results/postprocessed",
    L_beam=5.4,
    E=210e9,
    I=57.48e-8,
    rho=7850.0,
    A=65.423 * 0.0001,
    n_elem=540,
    output_type="csv",
    output_loc="C4S2",
    output_name="C4S2-MS-FNO_r2_after",
):
    """Compute R2 on the validation set after bias compensation."""
    interp_cfg = _load_cfg(interp_cfg_path)
    msfno_cfg = _load_cfg(msfno_cfg_path)
    resnet_cfg = _load_cfg(resnet_cfg_path)

    interp_model = _build_model_from_cfg(interp_cfg).to(DEVICE)
    interp_model.load_state_dict(
        torch.load(_load_first_pt(interp_cfg), map_location=DEVICE, weights_only=True)
    )
    interp_model.eval()

    pred_key = predictor.lower()
    if pred_key not in {"resnet", "msfno", "mosrnet"}:
        raise ValueError("predictor must be in {'resnet','msfno','mosrnet'}")
    if pred_key == "resnet":
        predictor_cfg = resnet_cfg
    elif pred_key == "msfno":
        predictor_cfg = msfno_cfg
    else:
        raise KeyError("the pred model must be msfno or resnet")

    pred_model = _build_model_from_cfg(predictor_cfg).to(DEVICE)
    pred_model.load_state_dict(
        torch.load(_load_first_pt(predictor_cfg), map_location=DEVICE, weights_only=True)
    )
    pred_model.eval()

    ds_path = f"./datasets/{data_name}/{valid_set}.pt"
    ds = torch.load(ds_path, map_location=DEVICE)
    in_ch = max(3, msfno_cfg["train"]["in_channels"], resnet_cfg["train"]["in_channels"])
    modes = ds["mode"][:, :in_ch, :]
    dmg = ds["dmg"].float()

    if down_idx is None:
        down_idx = np.linspace(0, modes.shape[-1] - 1, num=9, dtype=int)
    down_idx = np.asarray(down_idx, dtype=int)

    loader = DataLoader(TensorDataset(modes[:, :, down_idx].float(), dmg.float()), batch_size=1, shuffle=False)

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
        pred_int = pred_model(x_in_int)

    r2_metric = R2()
    results = []

    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            x_rec = interp_model(x)
            batch_size, _, rec_len = x_rec.shape
            grid = (
                torch.linspace(0.0, 1.0, steps=rec_len, device=DEVICE)
                .reshape(1, 1, -1)
                .expand(batch_size, 1, rec_len)
            )
            x_in = torch.cat([x_rec, grid], dim=1)

            if y.shape[-1] != rec_len:
                y_aligned = torch.nn.functional.interpolate(
                    y, size=rec_len, mode="linear", align_corners=False
                )
            else:
                y_aligned = y

            y_pred = pred_int - pred_model(x_in)
            r2_loss = r2_metric(
                y_pred.view(y_pred.shape[0], -1),
                1 - y_aligned.view(y_aligned.shape[0], -1),
            )
            results.append(r2_loss.detach().cpu().item())

    mean_r2 = np.mean(results)
    df = pd.DataFrame({"R2_each": results, "R2_mean": [mean_r2] * len(results)})

    save_dir = os.path.join(output_dir, output_loc, output_type)
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{output_name}.csv")
    df.to_csv(csv_path, index=False)

    print(f"Saved results to {csv_path}")
    return df


def C4S2_r2_onlypredmodel(
    interp_cfg_path,
    msfno_cfg_path,
    resnet_cfg_path,
    predictor="msfno",
    data_name="beamdi_num",
    valid_set="beamdi_num_v1000",
    output_dir="./results/postprocessed",
    output_type="csv",
    output_loc="C4S2",
    output_name="C4S2-MS-FNO_r2_ori",
):
    """Compute R2 for the predictor model on full-resolution inputs."""
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
        raise KeyError("the pred model must be msfno or resnet")

    pred_model = _build_model_from_cfg(predictor_cfg).to(DEVICE)
    pred_model.load_state_dict(
        torch.load(_load_first_pt(predictor_cfg), map_location=DEVICE, weights_only=True)
    )
    pred_model.eval()

    ds_path = f"./datasets/{data_name}/{valid_set}.pt"
    ds = torch.load(ds_path, map_location=DEVICE)
    modes = ds["mode"][:, :, :]
    dmg = ds["dmg"].float()

    loader = DataLoader(TensorDataset(modes.float(), dmg.float()), batch_size=1, shuffle=False)

    r2_metric = R2()
    results = []

    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            y_pred = pred_model(x)
            r2_loss = r2_metric(y_pred.view(y_pred.shape[0], -1), y.view(y.shape[0], -1))
            results.append(r2_loss.detach().cpu().item())

    mean_r2 = np.mean(results)
    df = pd.DataFrame({"R2_each": results, "R2_mean": [mean_r2] * len(results)})

    save_dir = os.path.join(output_dir, output_loc, output_type)
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{output_name}.csv")
    df.to_csv(csv_path, index=False)

    print(f"Saved results to {csv_path}")
    return df


DEFAULT_INTERP_CFG = "mosrnet-beamdi_num_t8000-run01-250814-164957.yaml"
DEFAULT_MSFNO_CFG = "msfno-beamdi_num_t8000-run01-250814-165002.yaml"
DEFAULT_RESNET_CFG = "resnet-beamdi_num_t8000-run01-250814-165006.yaml"
DEFAULT_DATA_NAME = "beamdi_num"
DEFAULT_VALID_SET = "beamdi_num_v1000"


def main():
    print(">>> Running MS-FNO R2 BEFORE bias compensation ...")
    df_msfno_before = C4S2_r2_before_biascompen(
        interp_cfg_path=DEFAULT_INTERP_CFG,
        msfno_cfg_path=DEFAULT_MSFNO_CFG,
        resnet_cfg_path=DEFAULT_RESNET_CFG,
        predictor="msfno",
        data_name=DEFAULT_DATA_NAME,
        valid_set=DEFAULT_VALID_SET,
        output_loc="C4S2",
        output_name="C4S2-MS-FNO_r2_before",
    )
    print(df_msfno_before.head())

    print(">>> Running MS-FNO R2 AFTER bias compensation ...")
    df_msfno_after = C4S2_r2_after_biascompen(
        interp_cfg_path=DEFAULT_INTERP_CFG,
        msfno_cfg_path=DEFAULT_MSFNO_CFG,
        resnet_cfg_path=DEFAULT_RESNET_CFG,
        predictor="msfno",
        data_name=DEFAULT_DATA_NAME,
        valid_set=DEFAULT_VALID_SET,
        output_loc="C4S2",
        output_name="C4S2-MS-FNO_r2_after",
    )
    print(df_msfno_after.head())

    print(">>> Running MS-FNO R2 ORIGINAL ...")
    df_msfno_ori = C4S2_r2_onlypredmodel(
        interp_cfg_path=DEFAULT_INTERP_CFG,
        msfno_cfg_path=DEFAULT_MSFNO_CFG,
        resnet_cfg_path=DEFAULT_RESNET_CFG,
        predictor="msfno",
        data_name=DEFAULT_DATA_NAME,
        valid_set=DEFAULT_VALID_SET,
        output_loc="C4S2",
        output_name="C4S2-MS-FNO_r2_ori",
    )
    print(df_msfno_ori.head())

    print(">>> Running ResNet R2 BEFORE bias compensation ...")
    df_resnet_before = C4S2_r2_before_biascompen(
        interp_cfg_path=DEFAULT_INTERP_CFG,
        msfno_cfg_path=DEFAULT_MSFNO_CFG,
        resnet_cfg_path=DEFAULT_RESNET_CFG,
        predictor="resnet",
        data_name=DEFAULT_DATA_NAME,
        valid_set=DEFAULT_VALID_SET,
        output_loc="C4S2",
        output_name="C4S2-ResNet_r2_before",
    )
    print(df_resnet_before.head())

    print(">>> Running ResNet R2 AFTER bias compensation ...")
    df_resnet_after = C4S2_r2_after_biascompen(
        interp_cfg_path=DEFAULT_INTERP_CFG,
        msfno_cfg_path=DEFAULT_MSFNO_CFG,
        resnet_cfg_path=DEFAULT_RESNET_CFG,
        predictor="resnet",
        data_name=DEFAULT_DATA_NAME,
        valid_set=DEFAULT_VALID_SET,
        output_loc="C4S2",
        output_name="C4S2-ResNet_r2_after",
    )
    print(df_resnet_after.head())

    print(">>> Running ResNet R2 ORIGINAL ...")
    df_resnet_ori = C4S2_r2_onlypredmodel(
        interp_cfg_path=DEFAULT_INTERP_CFG,
        msfno_cfg_path=DEFAULT_MSFNO_CFG,
        resnet_cfg_path=DEFAULT_RESNET_CFG,
        predictor="resnet",
        data_name=DEFAULT_DATA_NAME,
        valid_set=DEFAULT_VALID_SET,
        output_loc="C4S2",
        output_name="C4S2-ResNet_r2_ori",
    )
    print(df_resnet_ori.head())


if __name__ == "__main__":
    main()
