"""
Author: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
Date: 2025-08-12 18:06:11
LastEditors: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
LastEditTime: 2025-08-14 16:16:14
"""

import os
from datetime import datetime

from utilities.config_util import check_config_file, save_config_to_yaml


# ----------------------------- user parameters ----------------------------- #
description = "SHM Framework of MoSRNet and MS-FNO"
model_name = "mosrnet"

data_name = "beamdi_num"
subset_name = "beamdi_num_t8000"
if not subset_name:
    subset_name = data_name
valid_set = "beamdi_num_v1000"

epochs = 120
batch_size = 16
in_channels = 3

learning_rate = 0.001
weight_decay = 0.01

down_idx = [0, 68, 135, 203, 270, 337, 405, 473, 540]
gt_idx = 540

losses = [
    ["LpLoss", 1, [1, 1, 0, 0, 1, 1, 0, 0], {"d": 2, "p": 2, "size_average": True, "reduction": True}],
]

evaluations = [
    ["R2", [1, 1, 0, 0, 1, 1, 0, 0], {"size_average": True, "reduction": True}],
    ["MAE", [1, 1, 0, 0, 1, 1, 0, 0], {"size_average": True, "reduction": True}],
    ["MAPE", [1, 1, 0, 0, 1, 1, 0, 0], {"size_average": True, "reduction": True}],
]

# Stats flags order: std, cv, skewness, kurtosis, min, max, median, variance.

scheduler = {
    "scheduler": "ExpLRScheduler",
    "warmup_epochs": 20,
    "decay_rate": 0.97,
    "initial_ratio": 0.00001,
}

model_config = {
    "model": model_name,
    "para": {
        "dim1": 16,
        "dim2fct": 2,
        "inlen": len(down_idx),
        "outlen": gt_idx,
        "num_subnets": 3,
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
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "in_channels": in_channels,
        "down_idx": down_idx,
        "gt_idx": gt_idx,
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
