'''
Author: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
Date: 2025-08-13 15:14:27
LastEditors: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
LastEditTime: 2025-08-13 21:28:33
'''


import copy
import os
import platform
import psutil

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

try:
    import wandb
    wandb_loaded = True
except ImportError:
    wandb_loaded = False

from models.resnet import ResNet
from utilities.muon  import SingleDeviceMuonWithAuxAdam
from experiments.train_msfno_and_resnet import train_1d
from utilities.scheduler import ExpLRScheduler
from utilities import utilkit as kit

#%%
def run_train_1d(config, config_name, device, model_class, use_wandb=False, sweep=False):
    
    randomseed = config["randomseed"]
    kit.set_seed(randomseed)
    
    # 系统信息打印
    system_info = {
        'platform': platform.system(),
        'platform_release': platform.release(),
        'platform_version': platform.version(),
        'architecture': platform.machine(),
        'cpu': platform.processor(),
        'ram': f"{round(psutil.virtual_memory().total / (1024.0 **3))} GB"
    }
    if torch.cuda.is_available():
        system_info['gpu'] = torch.cuda.get_device_name(0)
        system_info['gpu_count'] = torch.cuda.device_count()
    print("### System and Hardware Specifications ###")
    for k, v in system_info.items():
        print(f"{k}: {v}")
    print("########################################")
    
    # Wandb初始化（如果使用）
    try:
        import wandb
        wandb_loaded = True
    except ImportError:
        wandb_loaded = False
    if wandb_loaded and use_wandb:
        from datetime import datetime
        now = datetime.now()
        date_time = now.strftime("%y%m%d%H%M")
        wandb.require("core")
        if sweep:
            run = wandb.init(name=f"{config['project']['case']}",
                             config=config,
                             reinit=True)
        else:
            run = wandb.init(project=f"{config['project']['description']}",
                             name=f"{config['project']['case']}_Muon",
                             config=config,
                             reinit=True
                             )
        wandb.config.update(system_info)
    else:
        run = None
        
    data_name = config["data"]["data"]
    traindata = config["data"]["subset"]
    validdata = config["data"]["validset"]
    
    train_dict = torch.load(f"./datasets/{data_name}/{traindata}.pt", map_location="cpu")
    train_dict = {k: v.to(device) for k, v in train_dict.items()}
    train_dataset = TensorDataset(train_dict['mode'], train_dict['dmg'])
    train_loader = DataLoader(train_dataset, batch_size=config["train"]["batch_size"], shuffle=True)
    
    valid_dict = torch.load(f"./datasets/{data_name}/{validdata}.pt", map_location="cpu")
    valid_dict = {k: v.to(device) for k, v in valid_dict.items()}
    valid_dataset = TensorDataset(valid_dict['mode'], valid_dict['dmg'])
    valid_loader = DataLoader(valid_dataset, batch_size=config["train"]["batch_size"], shuffle=False)
    
    # 模型初始化：参数依次为
    # (in_channels, mode_in_dim, freq_in_dim, embed_dim, fno_modes, fno_layers, out_channels)
    model = model_class(**config["model"]["para"]).to(device)
    
    total_params = sum(p.numel() for p in model.parameters()) 
    if wandb and use_wandb:
        wandb.config.update({"total_params": total_params})


    # optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["learning_rate"], weight_decay=config["train"]["weight_decay"])
    nonhidden_params = [*model.projection1.parameters(), *model.projection2.parameters()]
    nonhidden_params_set = set(nonhidden_params)
    hidden_weights = [p for p in model.parameters() if p.ndim >= 2 and p not in nonhidden_params_set]
    hidden_gains_biases = [p for p in model.parameters() if p.ndim < 2 and p not in nonhidden_params_set]
    param_groups = [
        dict(params=hidden_weights, use_muon=True,
            lr=config["train"]["learning_rate"], weight_decay=config["train"]["weight_decay"]),
        dict(params=hidden_gains_biases+nonhidden_params, use_muon=False,
            lr=config["train"]["learning_rate"], betas=(0.9, 0.95), weight_decay=config["train"]["weight_decay"]),
    ]
    optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
    scheduler = ExpLRScheduler(optimizer, warmup_epochs=20, decay_rate=0.96, initial_ratio=0)
    # scheduler = SinExpLRScheduler(optimizer, warmup_epochs=20, decay_rate=0.96, initial_ratio=0, sin_amplitude=0.1, sin_frequency=0.1)
    
    results_path = config["paths"]["results_path"]
    os.makedirs(results_path, exist_ok=True)
    config_intask = copy.deepcopy(config)
    if "status" in config_intask:
        del config_intask["status"]
    config_intask["system_info"] = system_info
    config_intask["model"]["total_params"] = total_params
    file_path = os.path.join(results_path, config_name)
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_intask, f, default_flow_style=False, allow_unicode=True)
    del file_path
    
    trained_model = train_1d(config,
                     config_name,
                     model,
                     device,
                     train_loader,
                     valid_loader,
                     optimizer,
                     scheduler,
                     results_path,
                     wandb=run,
                     ckpt=False,
                     wandb_loaded=wandb_loaded,
                     use_wandb=use_wandb,
                     use_tqdm=True,
                     calc_stats=True)

    return trained_model


#%%
model_classes = {"resnet":ResNet}

# config_name = "EXP1-MSFNO-BeamDI01_T8000-DD-250620-174154.yaml"

directory = "./configs" 
yaml_files = [f for f in os.listdir(directory) if f.endswith(('.yaml', '.yml')) and f.startswith('resnet')]

for config_name in yaml_files:
    with open(f"./configs/{config_name}", "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  

    trained_model = run_train_1d(config, config_name, device, model_classes[config["model"]["model"]], use_wandb=False, sweep=False)
