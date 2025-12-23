"""
Author: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
Date: 2025-08-12 18:06:11
LastEditors: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
LastEditTime: 2025-08-14 16:11:32
"""

import os
from datetime import datetime

from utilities.config_util import check_config_file, save_config_to_yaml


# ----------------------------- user parameters ----------------------------- #
description = "SHM Framework of MoSRNet and MS-FNO"
model_name = "msfno"

data_name = "beamdi_num"
subset_name = "beamdi_num_t8000"
if not subset_name:
    subset_name = data_name
valid_set = "beamdi_num_v1000"

losses = [
    ["LpLoss", 1, [1, 1, 0, 0, 1, 1, 0, 0], {"d": 2, "p": 2, "size_average": True, "reduction": True}],
]

evaluations = [
    ["R2", [1, 1, 0, 0, 1, 1, 0, 0], {"size_average": True, "reduction": True}],
    ["MAE", [1, 1, 0, 0, 1, 1, 0, 0], {"size_average": True, "reduction": True}],
    ["MAPE", [1, 1, 0, 0, 1, 1, 0, 0], {"size_average": True, "reduction": True}],
]

# Stats flags order: std, cv, skewness, kurtosis, min, max, median, variance.

epochs = 170
batch_size = 16
in_channels = 3
out_channels = 1

learning_rate = 0.001
weight_decay = 0.01

scheduler = {
    "scheduler": "ExpLRScheduler",
    "warmup_epochs": 20,
    "decay_rate": 0.975,
    "initial_ratio": 0,
}

model_config = {
    "model": model_name,
    "para": {
        "in_channels": in_channels + 1,
        "mode_length": 541,
        "embed_dim": 128,
        "fno_modes": 16,
        "fno_layers": 3,
        "out_channels": out_channels,
    },
}

random_seed = 114514


config = {
    "project": {"description": description},
    "data": {"data": data_name, "subset": subset_name, "validset": valid_set},
    "loss": {"losses": losses, "evaluations": evaluations},
    "train": {
        "epochs": epochs,
        "batch_size": batch_size,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
    },
    "scheduler": scheduler,
    "model": model_config,
    "randomseed": random_seed,
}


current_time = datetime.now()
gentime = current_time.strftime("%y%m%d-%H%M%S")

config_dir = "./configs"
exist, case_name = check_config_file(config_dir, config, config["model"]["model"], subset_name, gentime)

if not exist:
    project_name = f"{config['model']['model']}-{data_name}"
    config["project"]["project"] = project_name
    config["project"]["case"] = case_name[:-5]

    results_dir = os.path.join("./results/models", project_name)
    results_path = os.path.join(results_dir, case_name[:-5])
    config_path = os.path.join(config_dir, case_name)

    config["paths"] = {"config_path": config_path, "results_path": results_path}
    save_config_to_yaml(config, config_path)
