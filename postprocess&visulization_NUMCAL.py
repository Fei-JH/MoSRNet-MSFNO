import os
import torch
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import torch.nn.functional as F

from models.MSFNO import MSFNO
from models.Baselines import ResNet
from models.MBCNNSR import MBCNNSR
from models.FNOInterpNet import FNOInterpNet

from losses.LpLoss import LpLoss
from losses.RMSRE import RMSRE
from losses.MAE import MAE
from losses.MAPE import MAPE
from losses.R2 import R2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_classes = {
    "MSFNO": MSFNO,
    "ResNet": ResNet,
    "FNOInterpNet": FNOInterpNet,
    "MBCNNSR": MBCNNSR
}

intep_path = "EXP4-MBCNNSR-BeamDI02_T8000-RD-250724-185445.yaml"
MSFNO_path = "EXP1-MSFNO-BeamDI02_T8000-DD-250721-164537.yaml"
RESNET_path = "EXP1-ResNet-BeamDI02_T8000-DD-250721-170607.yaml"

data_name = "BeamDI02"
validdata = "BeamDI02_V1000"

with open(f"./configs/{intep_path}", "r") as f:
    intep_config = yaml.load(f, Loader=yaml.SafeLoader)
with open(f"./configs/{MSFNO_path}", "r") as f:
    MSFNO_config = yaml.load(f, Loader=yaml.SafeLoader)
with open(f"./configs/{RESNET_path}", "r") as f:
    RESNET_config = yaml.load(f, Loader=yaml.SafeLoader)

def load_model(config):
    model_class = model_classes[config["model"]["model"]]
    model = model_class(**config["model"]["para"]).to(device)
    model_dir = os.path.join(config["paths"]["results_path"], "model")
    pt_file = [f for f in os.listdir(model_dir) if f.endswith(".pt")][0]
    pt_path = os.path.join(model_dir, pt_file)
    state_dict = torch.load(pt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model

intep_model = load_model(intep_config)
MSFNO_model = load_model(MSFNO_config)
RESNET_model = load_model(RESNET_config)

intep_model.eval()
MSFNO_model.eval()
RESNET_model.eval()

valid_dict = torch.load(f"./datasets/{data_name}/{validdata}.pt", map_location=device)
valid_mode = valid_dict['mode'][:, :4, :]
down_idx = np.array([0, 68, 135, 203, 270, 337, 405, 473, 540], dtype=np.int32)
valid_mode_gt = valid_mode[:, :MSFNO_config["train"]["in_channels"], :]
valid_mode_down = valid_mode[:, :MSFNO_config["train"]["in_channels"], down_idx]
valid_dmg = valid_dict['dmg'].float()
N, in_channel, L = valid_mode_gt.shape

mae_metric = MAE(size_average=False, reduction=False)
mape_metric = MAPE(size_average=False, reduction=False)
lp_metric = LpLoss(size_average=False, reduction=False)
rmsre_metric = RMSRE(size_average=False, reduction=False)

def process(model, x, interp_model=None):
    with torch.no_grad():
        if interp_model is not None:
            out = interp_model(x.to(device))
        else:
            out = x.to(device)
        grid = torch.linspace(0, 1, out.shape[-1]).to(device).reshape(1, 1, -1)
        grid = grid.expand(out.shape[0], 1, out.shape[2])
        out = torch.cat((out, grid), dim=1)
        pred = model(out)
        # 自动修正pred到[N,1,L]
        if pred.ndim == 3 and pred.shape[1] != 1 and pred.shape[2] == 1:
            pred = pred.permute(0,2,1)  # [N,L,1]->[N,1,L]
        return pred

def interp_to_541_tensor(pred_tensor):
    """
    用PyTorch线性插值将[N, 1, 540]插值为[N, 1, 541]
    """
    if pred_tensor.shape[1] == 1 and pred_tensor.shape[2] == 540:
        return F.interpolate(pred_tensor, size=541, mode='linear', align_corners=True)
    elif pred_tensor.shape[1] == 540 and pred_tensor.shape[2] == 1:
        pred_tensor = pred_tensor.permute(0,2,1)
        return F.interpolate(pred_tensor, size=541, mode='linear', align_corners=True)
    elif pred_tensor.shape[2] == 541:
        return pred_tensor  # 已经是541点
    else:
        raise ValueError(f"interp_to_541_tensor input shape not match: {pred_tensor.shape}")

combos = {
    'MS-FNO':   (valid_mode_gt, MSFNO_model, None),
    'MBCNNSR + MS-FNO': (valid_mode_down, MSFNO_model, intep_model),
    'ResNet':    (valid_mode_gt, RESNET_model, None),
    'MBCNNSR + ResNet':(valid_mode_down, RESNET_model, intep_model)
}


if not os.path.exists('./postprocess'):
    os.makedirs('./postprocess')

# 容器用于合并所有结果
rows_sample = []
rows_elem = []
meanstd_dict = {}

for flag, (input_x, model, interp) in combos.items():
    print(f'Processing {flag} ...')
    mae_maps_tmp = []
    for i in range(input_x.shape[0]):
        x_single = input_x[i:i+1]
        gt_single = valid_dmg[i:i+1].to(device)
        if gt_single.ndim == 2:
            gt_single = gt_single.unsqueeze(1)
        elif gt_single.ndim == 3 and gt_single.shape[1] != 1:
            gt_single = gt_single.permute(0,2,1)
        pred_single = process(model, x_single, interp)
        if flag in ['MBCNNSR + MS-FNO', 'MBCNNSR + ResNet']:
            pred_single = interp_to_541_tensor(pred_single)
        if pred_single.ndim == 2:
            pred_single = pred_single.unsqueeze(1)
        elif pred_single.ndim == 3 and pred_single.shape[1] != 1 and pred_single.shape[2] == 1:
            pred_single = pred_single.permute(0,2,1)
        if pred_single.shape != gt_single.shape:
            raise ValueError(f"{flag} sample {i}: pred shape {pred_single.shape}, gt shape {gt_single.shape}")
        # 损失
        mae = mae_metric(pred_single, gt_single).mean().item()
        mape = mape_metric(pred_single, gt_single).mean().item()
        lploss = lp_metric(pred_single, gt_single).mean().item()
        rmsre = rmsre_metric(pred_single, gt_single).mean().item()
        # 逐样本记录
        rows_sample.append({
            'sample_idx': i,
            'case': flag,
            'MAE': mae,
            'MAPE': mape,
            'LpLoss': lploss,
            'RMSRE': rmsre
        })
        # 逐节点误差
        mae_nodes = torch.abs(pred_single - gt_single).squeeze().cpu().numpy()  # [L]
        mae_maps_tmp.append(mae_nodes)
        for j in range(mae_nodes.shape[0]):
            rows_elem.append({
                'sample_idx': i,
                'case': flag,
                'node_id': j,
                'MAE': mae_nodes[j]
            })
    # 记录本工况mean-std
    mae_mat = np.stack(mae_maps_tmp, axis=0)  # [N, L]
    mean_pernode = mae_mat.mean(axis=0)
    std_pernode = mae_mat.std(axis=0)
    meanstd_dict[flag] = (mean_pernode, std_pernode)

# 保存逐样本指标
df_all_samples = pd.DataFrame(rows_sample)
df_all_samples.to_csv('./postprocess/all_samples_metrics.csv', index=False)
# 保存逐节点误差
df_all_elems = pd.DataFrame(rows_elem)
df_all_elems.to_csv('./postprocess/all_elements_mae.csv', index=False)
# 统计各工况各指标 mean, min, max, CV
rows_stat = []
for flag in combos:
    df_flag = df_all_samples[df_all_samples['case'] == flag]
    for metric in ['MAE', 'MAPE', 'LpLoss', 'RMSRE']:
        vals = df_flag[metric].values
        mean = np.mean(vals)
        minv = np.min(vals)
        maxv = np.max(vals)
        cv = np.std(vals) / (mean + 1e-8)
        rows_stat.append({
            'case': flag,
            'metric': metric,
            'mean': mean,
            'min': minv,
            'max': maxv,
            'CV': cv
        })
df_stat = pd.DataFrame(rows_stat)
df_stat.to_csv('./postprocess/case_statistics.csv', index=False)
#%%
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
# 合并画图
plt.figure(figsize=(12, 8))
for flag in combos:
    mean_pernode, std_pernode = meanstd_dict[flag]
    L_now = mean_pernode.shape[0]
    x = np.linspace(0, 5400, L_now)
    plt.plot(x, mean_pernode, label=f'{flag} mean')
    # plt.fill_between(x, mean_pernode-std_pernode, mean_pernode+std_pernode, alpha=0.15)

plt.yscale('log')
plt.ylim(0.001, 1)
plt.xlabel('Node (mapped to length)')
plt.ylabel('MAE')
plt.xlim(0, 5400)
plt.xticks(np.linspace(0, 5400, num=9))

plt.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.20),  # 画框正上方，间隔适中
    ncol=2,                      # 两列
    columnspacing=1.0,
    frameon=False,
    fontsize=24,
    handlelength=2,
    handletextpad=1,
    borderaxespad=0,
    bbox_transform=plt.gca().transAxes,
    mode='expand'
)

plt.tight_layout()  # 给上方图例留空间
plt.savefig('./postprocess/all_cases_mae_profile.png', dpi=300)
plt.close()
# %%
#%%
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde
import numpy as np

# 数据选择
flag = 'MS-FNO'
df_flag = df_all_elems[df_all_elems['case'] == flag]

# 获取变量
x = df_flag['node_id'].values  # 横轴
y = df_flag['MAE'].values      # 纵轴（对数尺度）

# KDE 估计密度
xy = np.vstack([x, np.log10(y + 1e-10)])  # 避免 log(0)
kde = gaussian_kde(xy)

# 构建网格
xgrid = np.linspace(x.min(), x.max(), 200)
ygrid = np.linspace(np.log10(y.min()+1e-10), np.log10(y.max()+1e-10), 200)
X, Y = np.meshgrid(xgrid, ygrid)
positions = np.vstack([X.ravel(), Y.ravel()])

# 密度估计
Z = np.reshape(kde(positions), X.shape)

# 三维绘图
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', antialiased=True)
ax.set_xlabel('Node Index')
ax.set_ylabel('log10(MAE)')
ax.set_zlabel('Density')
ax.set_title(f'3D Density Surface of Node-wise MAE - {flag}')

plt.tight_layout()
plt.savefig(f'./postprocess/{flag}_mae_density_surface.png', dpi=300)
plt.close()