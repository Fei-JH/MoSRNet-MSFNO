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
description = "BeamDI02_V1000"
model = "MSFNO"
# Data parameters
data = "BeamDI02"
subset = "BeamDI02_T500"
validset = "BeamDI02_V1000"

if not subset:
    subset = data    
    
# Loss parameters
losses = [
            ["LpLoss",1,[1,1,0,0,1,1,0,0], {"d":2, "p":2, "size_average":True, "reduction":True}]
         ]

evaluations = [
                ["R2",[1,1,0,0,1,1,0,0], {"reduction":"mean", "epsilon":1e-8}],
                ["MAE",[1,1,0,0,1,1,0,0], {"reduction":"mean"}]
              ]
#(compute_std, compute_cv, compute_skewness, compute_kurtosis,compute_min, compute_max, compute_median, compute_variance)
#eg. ["R2",[1,1,0,0,1,1,0,0]] means compute R2 with compute_std, compute_cv, compute_min, compute_max

# Training parameters
epochs = 150
batch_size = 16
in_chan = 3  
out_chan = 1  

learning_rate = 0.001
weight_decay = 0.01

scheduler = {
            "scheduler"    :"ExpLRScheduler",
            "warmup_epochs": 20,
            "decay_rate"   : 0.9798,
            "initial_ratio": 0
            }

# Model parameters
model = {
        "model":model,
        "para":{
        "in_channels":in_chan+1,
        "mode_length": 541,
        "embed_dim":128,
        "fno_modes":16,
        "fno_layers":3,
        "out_channels":out_chan,
        }
        }

model_name = model["model"]

# Project parameters
tasktype = "DD" #DD for data-driven; PI for physic-informed

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
        "in_channels": in_chan,
        "out_channels": out_chan,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay
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
   