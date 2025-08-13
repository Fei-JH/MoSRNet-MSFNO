 # -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 17:46:00 2024 (JST)

@author: Jinghao FEI
"""

import yaml
import os
from datetime import datetime
from utilities.config_util import check_config_file
from utilities.config_util import save_config_to_yaml

#%%
description = "250714 MBCNNSR"
model = "MBCNNSR"
# Data parameters
data = "BeamDI02"
subset = "BeamDI02_T8000"
validset = "BeamDI02_V1000"

if not subset:
    subset = data    
    
# Training parameters
epochs = 150
batch_size = 16 
in_chan = 3 

learning_rate = 0.001
weight_decay = 0.01

uncertainty_lr = 0.001 # Learning rate for uncertainty weighting, if used

# Downsample indices
down_idx = [0, 68, 135, 203, 270, 337, 405, 473, 540] #[0, 135, 203, 270, 337, 405, 540] 
gt_idx = 540

# Loss parameters
losses = [
            ["LpLoss",1,[1,1,0,0,1,1,0,0], {"d":2, "p":2, "size_average":True, "reduction":True}],
            # ["TVLoss",1,[1,1,0,0,1,1,0,0], {"weight":0.2, "reduction":"mean"}],
            # ["MOLoss",1,[1,1,0,0,1,1,0,0], {"reduction":"mean"}],
            # ["SOLoss",1,[1,1,0,0,1,1,0,0], {"reduction":"mean"}],
            # ["MOLoss_AIRM",1,[1,1,0,0,1,1,0,0], {"reduction":"mean", "normalize_diag":True}],
            # ["SOLoss_AIRM",1,[1,1,0,0,1,1,0,0], {"reduction":"mean", "normalize_diag":True}],
            ["SCLoss",1,[1,1,0,0,1,1,0,0], {"model_path":r".\results\MSFNO-BeamDI02\EXP1-MSFNO-BeamDI02_T8000-DD-250721-164537\model\MSFNO-250721170758.pt", "grid_len":gt_idx, "device":"cuda"}],
         ]

evaluations = [
                ["R2",[1,1,0,0,1,1,0,0], {"reduction":"mean", "epsilon":1e-8}],
                ["MAE",[1,1,0,0,1,1,0,0], {"reduction":"mean"}],
                ["SCLoss",[1,1,0,0,1,1,0,0], {"model_path":r".\results\MSFNO-BeamDI02\EXP1-MSFNO-BeamDI02_T8000-DD-250721-164537\model\MSFNO-250721170758.pt", "grid_len":gt_idx, "device":"cuda"}]
              ]
#(compute_std, compute_cv, compute_skewness, compute_kurtosis,compute_min, compute_max, compute_median, compute_variance)
#eg. ["R2",[1,1,0,0,1,1,0,0]] means compute R2 with compute_std, compute_cv, compute_min, compute_max

scheduler = {
            "scheduler"    :"ExpLRScheduler",
            "warmup_epochs": 20,
            "decay_rate"   : 0.9798,
            "initial_ratio": 0.00001
            }

# Model parameters
# model = {
#         "model":model,
#         "para":{
#         "in_points":len(down_idx),
#         "out_points":gt_idx,
#         "width":64,
#         "modes":16,
#         "num_subnets":3
#         }
#         }
    
model = {
        "model":model,
        "para":
        {"dim1":16,
        "dim2fct":2,
        "inlen":len(down_idx),
        "outlen":gt_idx,
        "num_subnets":3
        }
        }    

model_name = model["model"]

# Project parameters
tasktype = "RD" #DD for data-driven; PI for physic-informed; MTL for multi-task learning; RD for random

randomseed=114514

#%%
config = {
    # Project parameters
    "project": {
        "description": description,
        "tasktype": tasktype
    },

    # Data parameters
    "data": {
        "data": data,
        "subset": subset,
        "validset": validset
    },

    # Loss parameters
    "loss": {
        "losses": losses,
        "evaluations": evaluations
    },

    # Training parameters
    "train": {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "in_channels": in_chan,
        "down_idx": down_idx,
        "gt_idx": gt_idx,
        "uncertainty_lr": uncertainty_lr  

    },

    # Scheduler parameters
    "scheduler": scheduler,

    # Model parameters
    "model": model,
    
    #Random seed
    "randomseed": randomseed
}

#%%


# Case name
current_time = datetime.now()
gentime = current_time.strftime('%y%m%d-%H%M%S')

config_dir = "./configs"

exist, case_name = check_config_file(config_dir, config, config["model"]['model'], subset, tasktype, gentime)

if exist:
    pass
else:
    project_name = f"{config['model']['model']}-{data}"
    config["project"]["project"] = project_name
    config["project"]["case"] = case_name[:-5]
    
    results_dir = os.path.join("./results", project_name)
    results_path = os.path.join(results_dir, case_name[:-5])
    os.makedirs(results_path,exist_ok=True)
    config_path = os.path.join(config_dir, case_name)
    
    config["paths"] = {"config_path":config_path,
                        "results_path":results_path}
     
    save_config_to_yaml(config, config_path)
   