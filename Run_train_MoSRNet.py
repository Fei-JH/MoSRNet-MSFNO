# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 17:51:30 2025 (JST)

@author: Jinghao FEI
"""
# Standard library imports
import copy
import os
import platform
import psutil
import numpy as np

# Third-party library imports
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# Attempting to import wandb (optional dependency)
try:
    import wandb
    wandb_loaded = True
except ImportError:
    wandb_loaded = False

# Custom module imports
from tqdm import tqdm
from models.MBCNNSR import MBCNNSR
from models.MSFNO import MSFNO
from models.Baselines import ResNet
from utilities.muon  import SingleDeviceMuonWithAuxAdam
from experiments.train_MBCNNSR import train_1d
from experiments.scheduler import ExpLRScheduler
from utilities import utilkit as kit
from utilities.batch_assemble_matrices import batch_assemble_matrices

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
    
    # check if wandb is available and initialize it if required
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
                             name=f"{config['project']['case']}_MuonMTL",
                             config=config,
                             reinit=True
                             )
        wandb.config.update(system_info)
    else:
        run = None
        
    data_name = config["data"]["data"]
    traindata = config["data"]["subset"]
    validdata = config["data"]["validset"] 

    down_idx = np.array(config["train"]["down_idx"], dtype=np.int32)

    use_MK = kit.need_MK_matrices(config)
    # train data
    train_dict = torch.load(f"./datasets/{data_name}/{traindata}.pt", map_location="cpu")
    train_mode = train_dict['mode'][:, :config["train"]["in_channels"], :]
    gt_idx = torch.linspace(0, train_mode.shape[2]-1, steps=config["train"]["gt_idx"]).round().long()
    train_mode_gt = train_mode[:, :, gt_idx]
    train_mode_down = train_mode[:, :, down_idx]
    train_dmg = kit.adaptive_average_downsample(train_dict['dmg'].float(), config["train"]["gt_idx"])
    train_dmg_matrics = kit.adaptive_average_downsample(train_dict['dmg'].float(), config["train"]["gt_idx"]-1)
    # train_dmg = train_dict['dmg'][:, :, gt_idx].float()
    print(f"train_dmg shape: {train_dmg.shape}, train_mode_down shape: {train_mode_down.shape}, train_mode_gt shape: {train_mode_gt.shape}")
   
    if use_MK:
        train_Mcond, train_Kcond = batch_assemble_matrices(
            dmgfield_tensor=train_dmg_matrics.squeeze(), n_target_elements=config["train"]["gt_idx"]-1,
            assemble_fn=kit.assemble_condensed_matrices,
            L=1, E=1, I=1, rho=1, A=1,
            batch_size=50, device="cpu", verbose=True
        )
    else:
        train_empty_M = torch.zeros(train_dmg.shape[0], 1, 1) 
        train_empty_K = torch.zeros_like(train_empty_M)
        train_Mcond, train_Kcond = train_empty_M, train_empty_K

    train_dataset = TensorDataset(train_mode_down.float(), 
                                train_mode_gt.float(), 
                                train_dmg.float(), 
                                train_Mcond.float(), 
                                train_Kcond.float()
                                )
    train_loader = DataLoader(train_dataset, batch_size=config["train"]["batch_size"], shuffle=True)

    # valid data
    valid_dict = torch.load(f"./datasets/{data_name}/{validdata}.pt", map_location="cpu")
    valid_mode = valid_dict['mode'][:, :config["train"]["in_channels"], :]
    valid_mode_gt = valid_mode[:, :, gt_idx]
    valid_mode_down = valid_mode[:, :, down_idx]
    valid_dmg = kit.adaptive_average_downsample(valid_dict['dmg'].float(), config["train"]["gt_idx"])
    valid_dmg_matrics = kit.adaptive_average_downsample(valid_dict['dmg'].float(), config["train"]["gt_idx"]-1)

    # valid_dmg = valid_dict['dmg'][:, :, gt_idx].float()
    
    if use_MK:
        valid_Mcond, valid_Kcond = batch_assemble_matrices(
            dmgfield_tensor=valid_dmg_matrics.squeeze(), n_target_elements=config["train"]["gt_idx"]-1,
            assemble_fn=kit.assemble_condensed_matrices,
            L=1, E=1, I=1, rho=1, A=1,
            batch_size=50, device="cpu", verbose=True
        )
    else:
        valid_empty_M = torch.zeros(valid_dmg.shape[0], 1, 1) 
        valid_empty_K = torch.zeros_like(valid_empty_M)
        valid_Mcond, valid_Kcond = valid_empty_M, valid_empty_K

    valid_dataset = TensorDataset(valid_mode_down.float(), 
                                valid_mode_gt.float(), 
                                valid_dmg.float(), 
                                valid_Mcond.float(), 
                                valid_Kcond.float()
                                )
    valid_loader = DataLoader(valid_dataset, batch_size=config["train"]["batch_size"], shuffle=False)


    # (dim1, dim2fct, num_subnets=3)
    model = model_class(**config["model"]["para"]).to(device)

    total_params = sum(p.numel() for p in model.parameters()) 
    if wandb and use_wandb:
        wandb.config.update({"total_params": total_params})

    # optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["learning_rate"], weight_decay=config["train"]["weight_decay"])
    nonhidden_params = []
    # for subnet in model.subnets:
    #     nonhidden_params.extend(list(subnet.projection1.parameters()))
    #     nonhidden_params.extend(list(subnet.projection2.parameters()))
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
    scheduler = ExpLRScheduler(optimizer, 
                               warmup_epochs=config["scheduler"]["warmup_epochs"], 
                               decay_rate=config["scheduler"]["decay_rate"], 
                               initial_ratio=config["scheduler"]["initial_ratio"])
    
    results_path = config["paths"]["results_path"]
    config_intask = copy.deepcopy(config)
    if "status" in config_intask:
        del config_intask["status"]
    config_intask["system_info"] = system_info
    config_intask["model"]["total_params"] = total_params
    file_path = os.path.join(results_path, config_name)
    file_path = kit.safe_path(file_path)
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
                     use_UW=False, 
                     calc_stats=False)

    return trained_model


#%%
model_classes = {
                "MBCNNSR":MBCNNSR,
                "PDUModesNet":PDUModesNet,
                "FNOInterpNet":FNOInterpNet
                }

# config_name = "EXP5-MBCNNSR-BeamDI02_T8000-RD-250724-174406.yaml"

directory = "./configs" 
yaml_files = [f for f in os.listdir(directory) if f.endswith(('.yaml', '.yml'))]

for config_name in yaml_files:
    with open(f"./configs/{config_name}", "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
    
    trained_model = run_train_1d(config, config_name, device, model_classes[config["model"]["model"]], use_wandb=True, sweep=False)
