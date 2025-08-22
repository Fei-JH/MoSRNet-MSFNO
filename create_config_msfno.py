'''
Author: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
Date: 2025-08-12 18:06:11
LastEditors: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
LastEditTime: 2025-08-14 16:11:32
'''


import os
from datetime import datetime
from utilities.config_util import check_config_file
from utilities.config_util import save_config_to_yaml

#%%
""" Input all the parameters in this block"""

description = "SHM Framework of MoSRNet and MS-FNO"
model = "msfno"
# Data parameters
data = "beamdi_num"
subset = "beamdi_num_t8000"
validset = "beamdi_num_v1000"

if not subset:
    subset = data    
    
# Loss parameters
losses = [
            ["LpLoss",1,[1,1,0,0,1,1,0,0], {"d":2, "p":2, "size_average":True, "reduction":True}]
         ]

evaluations = [
                ["R2",[1,1,0,0,1,1,0,0], {"size_average":True, "reduction":True}],
                ["MAE",[1,1,0,0,1,1,0,0], {"size_average":True, "reduction":True}],
                ["MAPE",[1,1,0,0,1,1,0,0], {"size_average":True, "reduction":True}]
              ]
#(compute_std, compute_cv, compute_skewness, compute_kurtosis,compute_min, compute_max, compute_median, compute_variance)
#eg. ["R2",[1,1,0,0,1,1,0,0]] means compute R2 with compute_std, compute_cv, compute_min, compute_max

# Training parameters
epochs = 170
batch_size = 16
in_chan = 3  
out_chan = 1  

learning_rate = 0.001
weight_decay = 0.01

scheduler = {
            "scheduler"    :"ExpLRScheduler",
            "warmup_epochs": 20,
            "decay_rate"   : 0.975,
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

randomseed=114514

#%%
config = {
    # Project parameters
    "project": {
        "description": description,
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

exist, case_name = check_config_file(config_dir, config, config["model"]['model'], subset, gentime)

if exist:
    pass
else:
    project_name = f"{config['model']['model']}-{data}"
    config["project"]["project"] = project_name
    config["project"]["case"] = case_name[:-5]
    
    results_dir = os.path.join(r"./results/models", project_name)
    results_path = os.path.join(results_dir, case_name[:-5])
    config_path = os.path.join(config_dir, case_name)
    
    config["paths"] = {"config_path":config_path,
                        "results_path":results_path}
     
    save_config_to_yaml(config, config_path)
   