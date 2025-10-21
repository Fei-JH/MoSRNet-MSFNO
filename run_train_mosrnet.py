'''
Author: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
Date: 2025-08-13 15:14:27
LastEditors: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
LastEditTime: 2025-08-26 17:18:32
'''

import copy
import os
import numpy as np

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

try:
    import wandb
    wandb_loaded = True
except ImportError:
    wandb_loaded = False

from models.mosrnet import MoSRNet
from utilities.muon  import SingleDeviceMuonWithAuxAdam
from experiments.train_mosrnet import train_1d
from utilities.scheduler import ExpLRScheduler
from utilities import utilkit as kit

#%%
def run_train_1d(config, config_name, device, model_class, use_wandb=False, sweep=False):
    
    randomseed = config["randomseed"]
    kit.set_seed(randomseed)
    
    system_info = kit.get_system_info()
    
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

    # train data
    train_dict = torch.load(f"./datasets/{data_name}/{traindata}.pt", map_location="cpu")
    train_mode = train_dict['mode'][:, :config["train"]["in_channels"], :]
    gt_idx = torch.linspace(0, train_mode.shape[2]-1, steps=config["train"]["gt_idx"]).round().long()
    train_mode_gt = train_mode[:, :, gt_idx]
    train_mode_down = train_mode[:, :, down_idx]
  
    train_dataset = TensorDataset(train_mode_down.float(), 
                                train_mode_gt.float(), 
                                )
    train_loader = DataLoader(train_dataset, batch_size=config["train"]["batch_size"], shuffle=True)

    # valid data
    valid_dict = torch.load(f"./datasets/{data_name}/{validdata}.pt", map_location="cpu")
    valid_mode = valid_dict['mode'][:, :config["train"]["in_channels"], :]
    valid_mode_gt = valid_mode[:, :, gt_idx]
    valid_mode_down = valid_mode[:, :, down_idx]

    valid_dataset = TensorDataset(valid_mode_down.float(), 
                                valid_mode_gt.float(), 
                                )
    valid_loader = DataLoader(valid_dataset, batch_size=config["train"]["batch_size"], shuffle=False)


    # (dim1, dim2fct, num_subnets=3)
    model = model_class(**config["model"]["para"]).to(device)

    total_params = sum(p.numel() for p in model.parameters()) 
    if wandb and use_wandb:
        wandb.config.update({"total_params": total_params})

    # optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["learning_rate"], weight_decay=config["train"]["weight_decay"])
    nonhidden_params = []
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
    
    results_path = kit.safe_path(config["paths"]["results_path"])
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
model_classes = {"mosrnet":MoSRNet}

if __name__ == "__main__":
    directory = "./configs"
    yaml_files = [
        f for f in os.listdir(directory)
        if f.endswith(('.yaml', '.yml')) and f.startswith('mosrnet')
    ]

    for config_name in yaml_files:
        with open(f"./configs/{config_name}", "r") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        trained_model = run_train_1d(
            config,
            config_name,
            device,
            model_classes[config["model"]["model"]],
            use_wandb=True,
            sweep=False

        )

