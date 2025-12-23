"""
Author: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
Date: 2025-08-12 18:06:19
LastEditors: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
LastEditTime: 2025-10-21 15:59:34
"""

import os
import random

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from models.mosrnet import MoSRNet
from models.msfno import MSFNO
from models.resnet import ResNet
from utilities.euler_bernoulli_beam_fem import BeamAnalysis
from utilities.mcs_util import generate_sequence, interpolate_1d


SEED = 114514
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ----------------------------- parameters ----------------------------- #
NUM_SIM = 8000
NUM_ELEM = 540
DOWN_IDX = np.array([0, 68, 135, 203, 270, 337, 405, 473, 540], dtype=np.int32)

L = 5.4
E = 210e9
I = 57.48e-8
RHO = 7850
A = 65.423 * 0.0001

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ----------------------------- load models ---------------------------- #
MODEL_CLASSES = {"msfno": MSFNO, "resnet": ResNet, "mosrnet": MoSRNet}

INTERP_CFG = "mosrnet-beamdi_num_t8000-run01-250814-164957.yaml"
MSFNO_CFG = "msfno-beamdi_num_t8000-run01-250814-165002.yaml"
RESNET_CFG = "resnet-beamdi_num_t8000-run01-250814-165006.yaml"

with open(f"./configs/{INTERP_CFG}", "r", encoding="utf-8") as f:
    interp_config = yaml.safe_load(f)
with open(f"./configs/{MSFNO_CFG}", "r", encoding="utf-8") as f:
    msfno_config = yaml.safe_load(f)
with open(f"./configs/{RESNET_CFG}", "r", encoding="utf-8") as f:
    resnet_config = yaml.safe_load(f)

# Load interpolation model.
interp_model = MODEL_CLASSES[interp_config["model"]["model"]](**interp_config["model"]["para"]).to(device)
interp_model_dir = os.path.join(interp_config["paths"]["results_path"], "model")
for file in os.listdir(interp_model_dir):
    if file.endswith(".pt"):
        interp_ckpt_path = os.path.join(interp_model_dir, file)
        break
state_dict = torch.load(interp_ckpt_path, map_location=device, weights_only=True)
interp_model.load_state_dict(state_dict)

# Load MSFNO model.
msfno_model = MODEL_CLASSES[msfno_config["model"]["model"]](**msfno_config["model"]["para"]).to(device)
msfno_model_dir = os.path.join(msfno_config["paths"]["results_path"], "model")
for file in os.listdir(msfno_model_dir):
    if file.endswith(".pt"):
        msfno_ckpt_path = os.path.join(msfno_model_dir, file)
        break
state_dict = torch.load(msfno_ckpt_path, map_location=device, weights_only=True)
msfno_model.load_state_dict(state_dict)

# Load ResNet model.
resnet_model = MODEL_CLASSES[resnet_config["model"]["model"]](**resnet_config["model"]["para"]).to(device)
resnet_model_dir = os.path.join(resnet_config["paths"]["results_path"], "model")
for file in os.listdir(resnet_model_dir):
    if file.endswith(".pt"):
        resnet_ckpt_path = os.path.join(resnet_model_dir, file)
        break
state_dict = torch.load(resnet_ckpt_path, map_location=device, weights_only=True)
resnet_model.load_state_dict(state_dict)

interp_model.eval()
msfno_model.eval()
resnet_model.eval()

# ------------------------- calculate intact state ------------------------- #
intact = np.ones(NUM_ELEM)
beam_intact = BeamAnalysis(L, E, I, RHO, A, NUM_ELEM)
beam_intact.assemble_matrices(dmgfield=intact, mass_dmg_power=0)
beam_intact.apply_BC()
_, eigenvectors = beam_intact.solve_eigenproblem()
u_vectors, _ = beam_intact.split_eigenvectors(eigenvectors)
modes_intact = u_vectors[:, 2:5].T
for i in range(modes_intact.shape[0]):
    if modes_intact[i, 14] < 0:
        modes_intact[i] = -modes_intact[i]
    max_abs = np.max(np.abs(modes_intact[i]))
    if max_abs > 0:
        modes_intact[i] = modes_intact[i] / max_abs

modes_intact_down_tensor = (
    torch.tensor(modes_intact[:, DOWN_IDX], dtype=torch.float32).to(device).unsqueeze(0)
)
with torch.no_grad():
    modes_intact_interp = interp_model(modes_intact_down_tensor).squeeze()

grid = np.linspace(0, 1, num=modes_intact_interp.shape[1], endpoint=True)
grid_tensor = torch.from_numpy(grid).float().to(device).unsqueeze(0)
modes_intact_interp_merged = torch.cat((modes_intact_interp, grid_tensor), dim=0).unsqueeze(0)

with torch.no_grad():
    msfno_intact_interp_pred = msfno_model(modes_intact_interp_merged)
    resnet_intact_interp_pred = resnet_model(modes_intact_interp_merged)

# ----------------------------- Monte Carlo ----------------------------- #
true_idx_list = []
msfno_idx_list = []
resnet_idx_list = []
msfno_int_idx_list = []
resnet_int_idx_list = []

for _ in tqdm(range(NUM_SIM), desc="Running Monte Carlo Simulation"):
    dmg, loc, _ = generate_sequence(length=NUM_ELEM, y=10, noise_range=0.05, dip_range=(0.15, 0.6))
    true_idx = loc

    beam = BeamAnalysis(L, E, I, RHO, A, NUM_ELEM)
    beam.assemble_matrices(dmgfield=dmg, mass_dmg_power=0)
    beam.apply_BC()
    _, eigenvectors = beam.solve_eigenproblem()
    u_vectors, _ = beam.split_eigenvectors(eigenvectors)
    modes = u_vectors[:, 2:5].T
    for i in range(modes.shape[0]):
        if modes[i, 14] < 0:
            modes[i] = -modes[i]
        max_abs = np.max(np.abs(modes[i]))
        if max_abs > 0:
            modes[i] = modes[i] / max_abs

    # Prepare model inputs.
    modes_down_tensor = torch.tensor(modes[:, DOWN_IDX], dtype=torch.float32).to(device).unsqueeze(0)
    with torch.no_grad():
        modes_interp = interp_model(modes_down_tensor).squeeze()
    grid = np.linspace(0, 1, num=modes_interp.shape[1], endpoint=True)
    grid_tensor = torch.from_numpy(grid).float().to(device).unsqueeze(0)
    modes_interp_merged = torch.cat((modes_interp, grid_tensor), dim=0).unsqueeze(0)
    modes_tensor = torch.tensor(modes, dtype=torch.float32).to(device).unsqueeze(0)
    grid_full = np.linspace(0, 1, num=modes_tensor.shape[2], endpoint=True)
    grid_tensor_full = torch.from_numpy(grid_full).float().to(device).unsqueeze(0)
    modes_merged = torch.cat((modes_tensor.squeeze(0), grid_tensor_full), dim=0).unsqueeze(0)

    # Predict damage using MSFNO and ResNet.
    with torch.no_grad():
        msfno_pred = msfno_model(modes_merged).squeeze().cpu().numpy()
        resnet_pred = resnet_model(modes_merged).squeeze().cpu().numpy()
        msfno_interp_pred = msfno_model(modes_interp_merged).squeeze().cpu().numpy()
        resnet_interp_pred = resnet_model(modes_interp_merged).squeeze().cpu().numpy()

    msfno_res = 1 - msfno_pred
    resnet_res = 1 - resnet_pred
    msfno_int_res = msfno_intact_interp_pred.squeeze().cpu().numpy() - msfno_interp_pred
    resnet_int_res = resnet_intact_interp_pred.squeeze().cpu().numpy() - resnet_interp_pred

    # Interpolate residuals to full length.
    msfno_res = interpolate_1d(msfno_res, NUM_ELEM)
    resnet_res = interpolate_1d(resnet_res, NUM_ELEM)
    msfno_int_res = interpolate_1d(msfno_int_res, NUM_ELEM)
    resnet_int_res = interpolate_1d(resnet_int_res, NUM_ELEM)

    msfno_idx = int(np.argmax(msfno_res))
    resnet_idx = int(np.argmax(resnet_res))
    msfno_int_idx = int(np.argmax(msfno_int_res))
    resnet_int_idx = int(np.argmax(resnet_int_res))

    true_idx_list.append(int(true_idx))
    msfno_idx_list.append(msfno_idx)
    resnet_idx_list.append(resnet_idx)
    msfno_int_idx_list.append(msfno_int_idx)
    resnet_int_idx_list.append(resnet_int_idx)

# ------------------------------ save results ------------------------------ #
os.makedirs("./results/mcs_test", exist_ok=True)
results_df = pd.DataFrame(
    {
        "true_idx": true_idx_list,
        "msfno_idx": msfno_idx_list,
        "resnet_idx": resnet_idx_list,
        "msfno_int_idx": msfno_int_idx_list,
        "resnet_int_idx": resnet_int_idx_list,
    }
)
results_df.to_csv("./results/mcs_test/msc_maxpos_rawidx.csv", index=False)
print("results saved to ./results/mcs_test/msc_maxpos_rawidx.csv")
