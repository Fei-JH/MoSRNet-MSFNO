
import copy
from http.cookiejar import LWPCookieJar
import os
import platform
from random import sample
import psutil
from datetime import datetime

# Third-party library imports
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path


# Custom module imports
from models.MSFNO import MSFNO
from models.Baselines import DNN, ResNet
from models.MBCNNSR import MBCNNSR
from models.FNOInterpNet import FNOInterpNet
from experiments.train_MSFNO import train_1d
from experiments.scheduler import ExpLRScheduler, SinExpLRScheduler
from utilities import utilkit as kit
#%%
model_classes = {
                "MSFNO":MSFNO,
                "ResNet":ResNet,
                "FNOInterpNet":FNOInterpNet,
                "MBCNNSR": MBCNNSR
                }

#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# EXP1-FNOInterpNet-BeamDI01_T8000-7S-250710-222026 
# EXP2-FNOInterpNet-BeamDI01_T8000-7S-250710-222034
# EXP3-FNOInterpNet-BeamDI01_T8000-7S-250710-222038
# EXP4-FNOInterpNet-BeamDI01_T8000-7S-250710-222042
#EXP1-FNOInterpNet-BeamDI01_T8000-RD-250716-152233

intep_path = "EXP4-MBCNNSR-BeamDI02_T8000-RD-250724-185445.yaml"
MSFNO_path = "EXP1-MSFNO-BeamDI02_T8000-DD-250721-164537.yaml"
RESNET_path = "EXP1-ResNet-BeamDI02_T8000-DD-250721-170607.yaml"

# intep_path = "EXP1-FNOInterpNet-BeamDI01_T8000-RD-250716-152233.yaml"
# MSFNO_path = "EXP1-MSFNO-BeamDI01_T8000-DD-250620-204614.yaml"
# RESNET_path = "EXP1-ResNet-BeamDI01_T8000-DD-250620-204628.yaml"


data_name = "BeamDI_EXP_old_short"
validdata = "SCUT"
baseline = "REIN"

resolution = 540
fig_name = f"{validdata}_MSFNO"
#%% Load configuration files
with open(f"./configs/{intep_path}", "r") as f:
        intep_config = yaml.load(f, Loader=yaml.SafeLoader)
with open(f"./configs/{MSFNO_path}", "r") as f:
        MSFNO_config = yaml.load(f, Loader=yaml.SafeLoader)
with open(f"./configs/{RESNET_path}", "r") as f:
        RESNET_config = yaml.load(f, Loader=yaml.SafeLoader)

# intep_config["model"]["para"]["out_points"] = resolution
#%%
model_class = model_classes[intep_config["model"]["model"]]
intep_model = model_class(**intep_config["model"]["para"]).to(device)
model_dir = os.path.join(intep_config["paths"]["results_path"],"model")
for file in os.listdir(model_dir):
    if file.endswith(".pt"):
        intep_path = os.path.join(model_dir, file)
        break
state_dict = torch.load(intep_path, map_location=device, weights_only=True)
intep_model.load_state_dict(state_dict)

model_class = model_classes[MSFNO_config["model"]["model"]]
MSFNO_model = model_class(**MSFNO_config["model"]["para"]).to(device)
model_dir = os.path.join(MSFNO_config["paths"]["results_path"],"model")
for file in os.listdir(model_dir):
    if file.endswith(".pt"):
        MSFNO_path = os.path.join(model_dir, file)
        break
state_dict = torch.load(MSFNO_path, map_location=device, weights_only=True)
MSFNO_model.load_state_dict(state_dict)

model_class = model_classes[RESNET_config["model"]["model"]]
RESNET_model = model_class(**RESNET_config["model"]["para"]).to(device)
model_dir = os.path.join(RESNET_config["paths"]["results_path"],"model")
for file in os.listdir(model_dir):
    if file.endswith(".pt"):
        RESNET_path = os.path.join(model_dir, file)
        break
state_dict = torch.load(RESNET_path, map_location=device, weights_only=True)
RESNET_model.load_state_dict(state_dict)
#%%
# valid data
valid_dict = torch.load(f"./datasets/{data_name}/{validdata}.pt", map_location=device)
valid_mode = valid_dict['mode'][:,:4,:]

down_idx = np.array(intep_config["train"]["down_idx"], dtype=np.int32)
gt_idx = torch.linspace(0, valid_mode.shape[2]-1, steps=intep_config["train"]["gt_idx"]).round().long()

valid_mode_gt = valid_mode[:, :MSFNO_config["train"]["in_channels"],:]
valid_mode_down = valid_mode[:, :MSFNO_config["train"]["in_channels"],:]
valid_dmg = valid_dict['dmg'].float()

valid_dataset = TensorDataset(valid_mode_down.float(), valid_mode_gt.float(), valid_dmg.float())
valid_loader = DataLoader(valid_dataset, batch_size=intep_config["train"]["batch_size"], shuffle=False)

# baseline data
baseline_dict = torch.load(f"./datasets/{data_name}/{baseline}.pt", map_location=device)
baseline_mode = baseline_dict['mode'][:,:4,:]

baseline_mode_gt = baseline_mode[:, :MSFNO_config["train"]["in_channels"],:]
baseline_mode_down = baseline_mode[:, :MSFNO_config["train"]["in_channels"],:]
baseline_dmg = baseline_dict['dmg'].float()

baseline_dataset = TensorDataset(baseline_mode_down.float(), baseline_mode_gt.float(), baseline_dmg.float())
baseline_loader = DataLoader(baseline_dataset, batch_size=intep_config["train"]["batch_size"], shuffle=False)
#%%
intep_model.eval()
MSFNO_model.eval()
RESNET_model.eval()
#%% Evaluation and visualization MSFNOence
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 24
matplotlib.rcParams['axes.labelsize'] = 24
matplotlib.rcParams['legend.fontsize'] = 24
matplotlib.rcParams['axes.titlesize'] = 24
matplotlib.rcParams['xtick.labelsize'] = 24
matplotlib.rcParams['ytick.labelsize'] = 24
#%%

# with torch.no_grad():
#     input = valid_mode[sample:sample+1]
#     output = MSFNO_model(input.to(device))
#     baseline = RESNET_model(input.to(device))
#     gt = valid_dmg[sample:sample+1].to(device)
#     output = output.squeeze().cpu()
#     gt = gt.squeeze().cpu()
#     baseline = baseline.squeeze().cpu()

#     plt.figure(figsize=(12, 6))
#     indices = np.arange(0, output.shape[-1])
#     plt.plot(indices, output.numpy(), label='Predicted Damage', color="blue", alpha=0.7)
#     plt.plot(indices, gt.numpy(), label='Ground Truth Damage', linestyle='--', color="grey", alpha=0.7)
#     plt.plot(indices, baseline.numpy(), label='Baseline Damage', linestyle=':', color="red", alpha=0.7)
#     plt.title(f'Sample {sample + 1}')
#     plt.xlabel('Spatial Dimension')
#     plt.ylabel('Damage')
#     plt.legend()
#     plt.show()
# %% Interpolation and visualization
# with torch.no_grad():
#     input = valid_mode_down[sample:sample+1]
#     output = intep_model(input.to(device))
#     gt = valid_mode_gt[sample:sample+1].to(device)
#     output = output.squeeze().cpu()
#     gt = gt.squeeze().cpu()

#     # 对每个通道进行maxabs归一化
#     # maxabs normalization: x_norm = x / max(abs(x))
#     # for ch in range(output.shape[0]):
#     #     maxabs = output[ch].abs().max()
#     #     print(f"Channel {ch+1} max absolute value: {maxabs}")
#     #     if maxabs > 0:
#     #         output[ch] = output[ch] / maxabs

#     num_channels = output.shape[0]
#     indices = np.arange(0, output.shape[-1])
#     fig, axs = plt.subplots(num_channels, 1, figsize=(12, 4 * num_channels), sharex=True)
#     gt_indices = np.linspace(0, output.shape[-1]-1, num=gt.shape[-1])
#     if num_channels == 1:
#         axs = [axs]
#     for ch in range(num_channels):
#         axs[ch].plot(indices, output[ch].numpy(), label='Predicted', color="blue", alpha=0.7)
#         axs[ch].plot(gt_indices, gt[ch].numpy(), label='Ground Truth', linestyle='--', color="grey", alpha=0.7)
#         axs[ch].set_title(f'Channel {ch+1}')
#         axs[ch].set_ylabel('Value')
#         axs[ch].legend()
#     axs[-1].set_xlabel('Spatial Dimension')
#     plt.tight_layout()
#     plt.show()

#%%iterp and infer
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerTuple
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import FuncFormatter

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerTuple

# Turn off gradient computation (inference only)
with torch.no_grad():
    # Get input sample and baseline
    input_sample = valid_mode_gt
    base_sample = baseline_mode_gt
    # Run interpolation model
    output1 = intep_model(input_sample.to(device))
    output2 = intep_model(base_sample.to(device))

    # Generate normalized grid and concatenate with outputs
    grid_idx = np.linspace(0, 1, num=output1.shape[-1], endpoint=True)
    grid = torch.from_numpy(grid_idx).float().to(device).reshape(1, 1, -1)
    # Expand grid to match batch and channel dimensions for concatenation
    grid_expanded = grid.expand(output1.shape[0], 1, output1.shape[2])
    mode_merged = torch.cat((output1, grid_expanded), dim=1)
    base_merged = torch.cat((output2, grid_expanded), dim=1)

    # Run MSFNO model
    sampleout = RESNET_model(mode_merged.to(device))
    baseout = RESNET_model(base_merged.to(device))

    sampleout = sampleout.mean(dim=0, keepdim=True)
    baseout = baseout.mean(dim=0, keepdim=True)

    # Postprocess outputs (inverse normalization to stiffness change, 0~100%)
    sampleout = 100 - sampleout.squeeze().cpu().numpy() * 100
    baseout = 100 - baseout.squeeze().cpu().numpy() * 100

    # Ground Truth
    samplegt = valid_dmg[0].to(device)
    basegt = baseline_dmg[0].to(device)
    
    samplegt = 100 - samplegt.squeeze().cpu().numpy() * 100
    basegt = 100 - basegt.squeeze().cpu().numpy() * 100

    # Calculate relative change
    output = sampleout - baseout
    gt = samplegt - basegt
    
    # Prepare x-axis indices
    num_points = output.shape[-1]
    x_indices = np.linspace(0, 5400, num=num_points)
    gt_indices = np.linspace(0, 5400, num=gt.shape[-1])

    # Separate positive/negative regions for color filling
    output = np.array(output)
    gt = np.array(gt)
    pos_mask = output > 0
    neg_mask = output <= 0

    # Create figure
    plt.figure(figsize=(12, 9))

    plt.plot(
        x_indices, output,
        color="#3A3A3A",
        lw=1,
        alpha=1.0,
        zorder=3,
        label='_nolegend_'
    )
    # Fill positive region
    plt.fill_between(
        x_indices, 0, output,
        where=pos_mask,
        facecolor='#EE7E77',
        alpha=1,
        label='_nolegend_',
        zorder=2
    )

    # Fill negative region
    plt.fill_between(
        x_indices, 0, output,
        where=neg_mask,
        facecolor='#68A7BE',
        alpha=1,
        label='_nolegend_',
        zorder=2
    )

    # Ground Truth (always black line, gray fill)
    plt.fill_between(
        gt_indices, 0, gt,
        facecolor='#CFCFCF',
        alpha=1.0,
        zorder=1,
        label='_nolegend_'
    )
    plt.step(
        gt_indices, gt,
        color="#3A3A3A",
        lw=1,
        linestyle='--',
        alpha=1.0,
        where='mid',       # 或 'pre'/'post'，根据你的需求选择
        zorder=4,
        label='_nolegend_'
    )

    # Draw black line at Y=0, on top
    plt.axhline(0, color='k', linewidth=1.5, zorder=10)

    # Axis settings
    plt.xlim(0, 5400)
    plt.xticks(np.linspace(0, 5400, num=9))
    plt.ylim(-60, 160)
    plt.yticks(np.linspace(-50, 150, num=9))

    # Custom Y-axis tick formatter
    def custom_yticks(y, pos):
        return f"{int(abs(y))}"
    plt.gca().yaxis.set_major_formatter(FuncFormatter(custom_yticks))

    # Region background color (positive/negative)
    plt.axhspan(0, 160, color='#feece7', alpha=1, zorder=0)
    plt.axhspan(-60, 0, color='#deeeed', alpha=1, zorder=0)

    # Label text in regions
    plt.text(150, 138, 'Stiffness Loss', color='#EE7E77', va='center', ha='left', fontsize=28, fontweight='bold')
    plt.text(150, -38, 'Stiffness Increase', color='#68A7BE', va='center', ha='left', fontsize=28, fontweight='bold')

    plt.ylabel('Stiffness Change (%)')
    plt.xlabel('Beam Span (mm)')

    # Legend: two colored patches (tuple) and ground truth
    red_patch = mpatches.Patch(color='#EE7E77', edgecolor='none')
    blue_patch = mpatches.Patch(color='#68A7BE', edgecolor='none')
    fill_patch = mpatches.Patch(facecolor='#CFCFCF', edgecolor='none')
    plt.legend(
        handles=[(red_patch, blue_patch), fill_patch],
        labels=["ResNet's Prediction", "Theoretical Damage Pattern"],
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc='best'
    )

    plt.grid(True, linestyle='-', alpha=0.3, zorder=0)
    plt.tight_layout()
    # plt.show()

    plt.savefig(f"./postprocess/{fig_name}.png", dpi=300)

