"""
Author: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
Date: 2025-08-13 15:14:27
LastEditors: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
LastEditTime: 2025-08-14 13:41:21
"""

import copy
import os

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

from experiments.train_msfno_and_resnet import train_1d
from models.msfno import MSFNO
from utilities.muon import SingleDeviceMuonWithAuxAdam
from utilities.scheduler import ExpLRScheduler
from utilities import utilkit as kit


def run_train_1d(config, config_name, device, model_class, use_wandb=False, sweep=False):
    """Run MSFNO training with a single config file."""
    random_seed = config["randomseed"]
    kit.set_seed(random_seed)

    system_info = kit.get_system_info()

    run = None
    if WANDB_AVAILABLE and use_wandb:
        wandb.require("core")
        if sweep:
            run = wandb.init(name=f"{config['project']['case']}", config=config, reinit=True)
        else:
            run = wandb.init(
                project=f"{config['project']['description']}",
                name=f"{config['project']['case']}_Muon",
                config=config,
                reinit=True,
            )
        wandb.config.update(system_info)

    data_name = config["data"]["data"]
    train_subset = config["data"]["subset"]
    valid_subset = config["data"]["validset"]

    train_dict = torch.load(f"./datasets/{data_name}/{train_subset}.pt", map_location="cpu")
    train_dict = {k: v.to(device) for k, v in train_dict.items()}
    train_dataset = TensorDataset(train_dict["mode"], train_dict["dmg"])
    train_loader = DataLoader(
        train_dataset, batch_size=config["train"]["batch_size"], shuffle=True
    )

    valid_dict = torch.load(f"./datasets/{data_name}/{valid_subset}.pt", map_location="cpu")
    valid_dict = {k: v.to(device) for k, v in valid_dict.items()}
    valid_dataset = TensorDataset(valid_dict["mode"], valid_dict["dmg"])
    valid_loader = DataLoader(
        valid_dataset, batch_size=config["train"]["batch_size"], shuffle=False
    )

    # Model initialization.
    model = model_class(**config["model"]["para"]).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    if WANDB_AVAILABLE and use_wandb:
        wandb.config.update({"total_params": total_params})

    nonhidden_params = [*model.projection1.parameters(), *model.projection2.parameters()]
    nonhidden_params_set = set(nonhidden_params)
    hidden_weights = [p for p in model.parameters() if p.ndim >= 2 and p not in nonhidden_params_set]
    hidden_gains_biases = [p for p in model.parameters() if p.ndim < 2 and p not in nonhidden_params_set]
    param_groups = [
        dict(
            params=hidden_weights,
            use_muon=True,
            lr=config["train"]["learning_rate"],
            weight_decay=config["train"]["weight_decay"],
        ),
        dict(
            params=hidden_gains_biases + nonhidden_params,
            use_muon=False,
            lr=config["train"]["learning_rate"],
            betas=(0.9, 0.95),
            weight_decay=config["train"]["weight_decay"],
        ),
    ]
    optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
    scheduler = ExpLRScheduler(optimizer, warmup_epochs=20, decay_rate=0.96, initial_ratio=0)

    results_path = config["paths"]["results_path"]
    os.makedirs(results_path, exist_ok=True)
    config_intask = copy.deepcopy(config)
    if "status" in config_intask:
        del config_intask["status"]
    config_intask["system_info"] = system_info
    config_intask["model"]["total_params"] = total_params
    config_path = os.path.join(results_path, config_name)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_intask, f, default_flow_style=False, allow_unicode=True)

    trained_model = train_1d(
        config,
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
        wandb_loaded=WANDB_AVAILABLE,
        use_wandb=use_wandb,
        use_tqdm=True,
        calc_stats=True,
    )

    return trained_model


model_classes = {"msfno": MSFNO}


if __name__ == "__main__":
    directory = "./configs"
    yaml_files = [
        f
        for f in os.listdir(directory)
        if f.endswith((".yaml", ".yml")) and f.startswith("msfno")
    ]

    for config_name in yaml_files:
        with open(f"./configs/{config_name}", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        run_train_1d(
            config,
            config_name,
            device,
            model_classes[config["model"]["model"]],
            use_wandb=False,
            sweep=False,
        )
