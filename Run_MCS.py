'''
Author: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
Date: 2025-08-12 18:06:19
LastEditors: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
LastEditTime: 2025-08-22 16:36:46
'''


import os
import numpy as np
import torch
from tqdm import tqdm
import random
import pandas as pd
import yaml

from utilities.mcs_util import generate_sequence, interpolate_1d
from utilities.euler_bernoulli_beam_fem import BeamAnalysis

from models.msfno import MSFNO
from models.resnet import ResNet
from models.mosrnet import MoSRNet

# ===================== 1. 固定随机种子 =====================
SEED = 114514
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ===================== 2. 参数设置 =====================
n_sim = 8000                # 蒙特卡洛样本数
n_elem = 540                # 单元总数
down_idx = np.array([0, 68, 135, 203, 270, 337, 405, 473, 540], dtype=np.int32) # 下采样位置

L = 5.4
E = 210e9
I = 57.48e-8
rho = 7850
A = 65.423 * 0.0001

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ===================== 3. 模型加载 =====================
MODEL_CLASSES = {"msfno": MSFNO, "resnet": ResNet, "mosrnet": MoSRNet}

INTEP_CFG  = "mosrnet-beamdi_num_t8000-run01-250814-164957.yaml"  
MSFNO_CFG  = "msfno-beamdi_num_t8000-run01-250814-165002.yaml"   
RESNET_CFG = "resnet-beamdi_num_t8000-run01-250814-165006.yaml"

with open(f"./configs/{INTEP_CFG}", "r") as f:
    intep_config = yaml.load(f, Loader=yaml.SafeLoader)
with open(f"./configs/{MSFNO_CFG}", "r") as f:
    msfno_config = yaml.load(f, Loader=yaml.SafeLoader)
with open(f"./configs/{RESNET_CFG}", "r") as f:
    resnet_config = yaml.load(f, Loader=yaml.SafeLoader)

# 加载intep模型
model_class = MODEL_CLASSES[intep_config["model"]["model"]]
intep_model = model_class(**intep_config["model"]["para"]).to(device)
model_dir = os.path.join(intep_config["paths"]["results_path"], "model")
for file in os.listdir(model_dir):
    if file.endswith(".pt"):
        intep_ckpt_path = os.path.join(model_dir, file)
        break
state_dict = torch.load(intep_ckpt_path, map_location=device, weights_only=True)
intep_model.load_state_dict(state_dict)

# 加载MSFNO模型
model_class = MODEL_CLASSES[msfno_config["model"]["model"]]
msfno_model = model_class(**msfno_config["model"]["para"]).to(device)
model_dir = os.path.join(msfno_config["paths"]["results_path"], "model")
for file in os.listdir(model_dir):
    if file.endswith(".pt"):
        msfno_ckpt_path = os.path.join(model_dir, file)
        break
state_dict = torch.load(msfno_ckpt_path, map_location=device, weights_only=True)
msfno_model.load_state_dict(state_dict)

# 加载ResNet模型
model_class = MODEL_CLASSES[resnet_config["model"]["model"]]
resnet_model = model_class(**resnet_config["model"]["para"]).to(device)
model_dir = os.path.join(resnet_config["paths"]["results_path"], "model")
for file in os.listdir(model_dir):
    if file.endswith(".pt"):
        resnet_ckpt_path = os.path.join(model_dir, file)
        break
state_dict = torch.load(resnet_ckpt_path, map_location=device, weights_only=True)
resnet_model.load_state_dict(state_dict)

intep_model.eval()
msfno_model.eval()
resnet_model.eval()

# ===================== 4. 计算intact工况的插值输出 =====================
intact = np.ones(n_elem)
beam_intact = BeamAnalysis(L, E, I, rho, A, n_elem)
beam_intact.assemble_matrices(dmgfield=intact, mass_dmg_power=0)
beam_intact.apply_BC()
_, eigenvectors = beam_intact.solve_eigenproblem()
u_vectors, _ = beam_intact.split_eigenvectors(eigenvectors)
modes_intact = u_vectors[:, 2:5].T
for i in range(modes_intact.shape[0]):
    if modes_intact[i, 14] < 0:
        modes_intact[i] = -modes_intact[i]
    maxabs = np.max(np.abs(modes_intact[i]))
    if maxabs > 0:
        modes_intact[i] = modes_intact[i] / maxabs

modes_intact_down_tensor = torch.tensor(modes_intact[:, down_idx], dtype=torch.float32).to(device).unsqueeze(0)
with torch.no_grad():
    modes_intact_intep = intep_model(modes_intact_down_tensor).squeeze()

grid = np.linspace(0, 1, num=modes_intact_intep.shape[1], endpoint=True)
grid_tensor = torch.from_numpy(grid).float().to(device).unsqueeze(0)
modes_intact_intep_merged = torch.cat((modes_intact_intep, grid_tensor), dim=0).unsqueeze(0)

modes_intact_tensor = torch.tensor(modes_intact, dtype=torch.float32).to(device)
grid_full = np.linspace(0, 1, num=modes_intact_tensor.shape[1], endpoint=True)
grid_tensor_full = torch.from_numpy(grid_full).float().to(device).unsqueeze(0)
modes_intact_merged = torch.cat((modes_intact_tensor, grid_tensor_full), dim=0).unsqueeze(0)

with torch.no_grad():
    msfno_intact_intep_pred = msfno_model(modes_intact_intep_merged)
    resnet_intact_intep_pred = resnet_model(modes_intact_intep_merged)

# ===================== 5. 蒙特卡洛仿真主循环 =====================
true_idx_list = []
msfno_idx_list = []
resnet_idx_list = []
msfno_int_idx_list = []
resnet_int_idx_list = []


for _ in tqdm(range(n_sim), desc="Running Monte Carlo Simulation"):
    # 1) 生成损伤场及真值
    dmg, loc, dgr = generate_sequence(length=n_elem, y=10, noise_range=0.05, dip_range=(0.15, 0.6))
    true_idx = loc

    # 2) 模态特征提取及归一化
    beam = BeamAnalysis(L, E, I, rho, A, n_elem)
    beam.assemble_matrices(dmgfield=dmg, mass_dmg_power=0)
    beam.apply_BC()
    _, eigenvectors = beam.solve_eigenproblem()
    u_vectors, _ = beam.split_eigenvectors(eigenvectors)
    modes = u_vectors[:, 2:5].T
    for i in range(modes.shape[0]):
        if modes[i, 14] < 0:
            modes[i] = -modes[i]
        maxabs = np.max(np.abs(modes[i]))
        if maxabs > 0:
            modes[i] = modes[i] / maxabs

    # 3) 输入准备
    modes_down_tensor = torch.tensor(modes[:, down_idx], dtype=torch.float32).to(device).unsqueeze(0)
    with torch.no_grad():
        modes_intep = intep_model(modes_down_tensor).squeeze()
    grid = np.linspace(0, 1, num=modes_intep.shape[1], endpoint=True)
    grid_tensor = torch.from_numpy(grid).float().to(device).unsqueeze(0)
    modes_intep_merged = torch.cat((modes_intep, grid_tensor), dim=0).unsqueeze(0)
    modes_tensor = torch.tensor(modes, dtype=torch.float32).to(device).unsqueeze(0)
    grid_full = np.linspace(0, 1, num=modes_tensor.shape[2], endpoint=True)
    grid_tensor_full = torch.from_numpy(grid_full).float().to(device).unsqueeze(0)
    modes_merged = torch.cat((modes_tensor.squeeze(0), grid_tensor_full), dim=0).unsqueeze(0)

    # 4) 模型推理
    with torch.no_grad():
        msfno_pred = msfno_model(modes_merged).squeeze().cpu().numpy()
        resnet_pred = resnet_model(modes_merged).squeeze().cpu().numpy()
        msfno_intep_pred = msfno_model(modes_intep_merged).squeeze().cpu().numpy()
        resnet_intep_pred = resnet_model(modes_intep_merged).squeeze().cpu().numpy()

    msfno_res = 1 - msfno_pred
    resnet_res = 1 - resnet_pred
    msfno_int_res = msfno_intact_intep_pred.squeeze().cpu().numpy() - msfno_intep_pred
    resnet_int_res = resnet_intact_intep_pred.squeeze().cpu().numpy() - resnet_intep_pred


    # Example usage
    msfno_res = interpolate_1d(msfno_res, n_elem)
    resnet_res = interpolate_1d(resnet_res, n_elem)
    msfno_int_res = interpolate_1d(msfno_int_res, n_elem)
    resnet_int_res = interpolate_1d(resnet_int_res, n_elem)

    msfno_idx = int(np.argmax(msfno_res))
    resnet_idx = int(np.argmax(resnet_res))
    msfno_int_idx = int(np.argmax(msfno_int_res))
    resnet_int_idx = int(np.argmax(resnet_int_res))

    true_idx_list.append(int(true_idx))
    msfno_idx_list.append(msfno_idx)
    resnet_idx_list.append(resnet_idx)
    msfno_int_idx_list.append(msfno_int_idx)
    resnet_int_idx_list.append(resnet_int_idx)

# ===================== 6. 保存csv结果 =====================
os.makedirs("./results/mcs_test", exist_ok=True)
results_df = pd.DataFrame({
    "true_idx": true_idx_list,
    "msfno_idx": msfno_idx_list,
    "resnet_idx": resnet_idx_list,
    "msfno_int_idx": msfno_int_idx_list,
    "resnet_int_idx": resnet_int_idx_list
})
results_df.to_csv("./results/mcs_test/msc_maxpos_rawidx.csv", index=False)
print("结果已保存")
