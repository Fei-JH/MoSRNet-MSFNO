"""
Author: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
Date: 2025-08-12 18:06:31
LastEditors: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
LastEditTime: 2025-08-14 16:49:46
"""

import csv
import os
import time
from datetime import datetime

import numpy as np
import torch
import yaml
from tqdm import tqdm

from utilities import utilkit as kit
from utilities.config_util import update_training_status
from utilities.train_util import compute_dataset_stats


def train_1d(
    config,
    config_name,
    model,
    device,
    train_loader,
    valid_loader,
    optimizer,
    scheduler,
    save_path,
    wandb,
    ckpt=True,
    wandb_loaded=False,
    use_wandb=False,
    use_tqdm=True,
    calc_stats=True,
):
    """Train a 1D model and record metrics per epoch."""
    update_training_status(config, phase="start")

    start_dt = datetime.now()
    timestamp = start_dt.strftime("%y%m%d%H%M%S")
    model_prefix = f"{config['model']['model']}-{timestamp}"

    with open(config["paths"]["config_path"], "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    loss_path = os.path.join(save_path, "loss")
    model_path = os.path.join(save_path, "model")
    os.makedirs(loss_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    epochs = range(1, config["train"]["epochs"] + 1)

    loss_names = [loss[0] for loss in config["loss"]["losses"]]
    loss_weights = [loss[1] for loss in config["loss"]["losses"]]
    eval_names = [evaluation[0] for evaluation in config["loss"]["evaluations"]]

    batch_losses = []
    batch_evaluations = []

    for loss in config["loss"]["losses"]:
        loss_ctor = kit.load_loss_function(f"losses.{loss[0].lower()}.{loss[0]}")
        batch_losses.append(loss_ctor(**loss[-1]))

    for evaluation in config["loss"]["evaluations"]:
        eval_ctor = kit.load_loss_function(f"losses.{evaluation[0].lower()}.{evaluation[0]}")
        batch_evaluations.append(eval_ctor(**evaluation[-1]))

    history = {
        "T_loss": [],
        "V_loss": [],
        "train_time": [],
        "epoch_time": [],
    }

    for name in loss_names:
        history[f"Tloss_{name}"] = {"mean": []}
        history[f"Vloss_{name}"] = {"mean": []}
        history[f"WTloss_{name}"] = []
        history[f"WVloss_{name}"] = []

    for name in eval_names:
        history[f"Teval_{name}"] = {"mean": []}
        history[f"Veval_{name}"] = {"mean": []}

    if calc_stats:
        loss_stats_flags = [loss[2] for loss in config["loss"]["losses"]]
        eval_stats_flags = [evaluation[1] for evaluation in config["loss"]["evaluations"]]

        sample_losses = []
        sample_evaluations = []

        for name in loss_names:
            loss_ctor = kit.load_loss_function(f"losses.{name}.{name}")
            sample_losses.append(loss_ctor(size_average=False, reduction=False))

        for name in eval_names:
            eval_ctor = kit.load_loss_function(f"losses.{name}.{name}")
            sample_evaluations.append(eval_ctor(size_average=False, reduction=False))

    for epoch in epochs:
        train_evals = np.zeros(len(batch_evaluations))
        valid_evals = np.zeros(len(batch_evaluations))
        train_losses = np.zeros(len(batch_losses))
        valid_losses = np.zeros(len(batch_losses))
        weighted_train_loss = np.zeros(1)
        weighted_valid_loss = np.zeros(1)
        loss_weights_tensor = torch.tensor(loss_weights, device=device)

        ep_str = f"[ EPOCH {epoch:03d} ]"
        total_len = 80
        side_len = (total_len - len(ep_str)) // 2
        extra = (total_len - len(ep_str)) % 2
        print("=" * side_len + ep_str + "=" * (side_len + extra))

        t1 = time.time()
        model.train()

        train_iter = train_loader
        if use_tqdm:
            bar_format = "{desc}|{bar}| Speed:{rate_fmt} {postfix}"
            train_iter = tqdm(
                train_loader,
                desc=f"[Epoch {epoch}] Training ",
                leave=True,
                ncols=80,
                bar_format=bar_format,
            )
            total_batch_time = 0.0

        for mode, dmg in train_iter:
            if use_tqdm:
                train_t = time.time()
            mode, dmg = mode.to(device), dmg.to(device)

            optimizer.zero_grad()
            out = model(mode)

            batch_loss_values = torch.empty(len(batch_losses)).to(device)
            batch_eval_values = torch.empty(len(batch_evaluations)).to(device)

            for idx, batch_evaluation in enumerate(batch_evaluations):
                batch_eval_values[idx] = batch_evaluation(
                    out.view(out.shape[0], -1), dmg.view(dmg.shape[0], -1)
                )
            for idx, batch_loss in enumerate(batch_losses):
                batch_loss_values[idx] = batch_loss(
                    out.view(out.shape[0], -1), dmg.view(dmg.shape[0], -1)
                )

            weighted_batch_losses = batch_loss_values * loss_weights_tensor
            weighted_batch_loss = torch.sum(weighted_batch_losses)

            weighted_batch_loss.backward()
            optimizer.step()

            train_losses += batch_loss_values.detach().cpu().numpy()
            weighted_train_loss += weighted_batch_loss.detach().cpu().numpy()
            train_evals += batch_eval_values.detach().cpu().numpy()

            if use_tqdm:
                batch_time = time.time() - train_t
                total_batch_time += batch_time
                train_iter.set_postfix_str(f"Total:{total_batch_time:7.3f} s")

        scheduler.step()
        time1 = time.time() - t1

        model.eval()
        with torch.no_grad():
            valid_iter = valid_loader
            if use_tqdm:
                bar_format = "{desc}|{bar}| Speed:{rate_fmt} {postfix}"
                valid_iter = tqdm(
                    valid_loader,
                    desc=f"[Epoch {epoch}] Validation",
                    leave=True,
                    ncols=80,
                    bar_format=bar_format,
                )
                total_valid_time = 0.0
            for mode, dmg in valid_iter:
                if use_tqdm:
                    valid_t = time.time()
                mode, dmg = mode.to(device, non_blocking=True), dmg.to(device, non_blocking=True)
                out = model(mode)

                batch_loss_values = torch.empty(len(batch_losses)).to(device)
                batch_eval_values = torch.empty(len(batch_evaluations)).to(device)

                for idx, batch_evaluation in enumerate(batch_evaluations):
                    batch_eval_values[idx] = batch_evaluation(
                        out.view(out.shape[0], -1), dmg.view(dmg.shape[0], -1)
                    )
                for idx, batch_loss in enumerate(batch_losses):
                    batch_loss_values[idx] = batch_loss(
                        out.view(out.shape[0], -1), dmg.view(dmg.shape[0], -1)
                    )

                weighted_batch_losses = batch_loss_values * loss_weights_tensor
                weighted_batch_loss = torch.sum(weighted_batch_losses)

                valid_losses += batch_loss_values.detach().cpu().numpy()
                weighted_valid_loss += weighted_batch_loss.detach().cpu().numpy()
                valid_evals += batch_eval_values.detach().cpu().numpy()

                if use_tqdm:
                    batch_time = time.time() - valid_t
                    total_valid_time += batch_time
                    valid_iter.set_postfix_str(f"Total:{total_valid_time:7.3f} s")

        len_train_loader = len(train_loader)
        len_valid_loader = len(valid_loader)
        train_evals /= len_train_loader
        valid_evals /= len_valid_loader
        train_losses /= len_train_loader
        weighted_train_loss /= len_train_loader
        valid_losses /= len_valid_loader
        weighted_valid_loss /= len_valid_loader

        history["T_loss"].append(weighted_train_loss.item())
        history["V_loss"].append(weighted_valid_loss.item())
        history["train_time"].append(time1)

        for idx, name in enumerate(loss_names):
            history[f"Tloss_{name}"]["mean"].append(train_losses[idx])
            history[f"Vloss_{name}"]["mean"].append(valid_losses[idx])
            history[f"WTloss_{name}"].append(train_losses[idx] * loss_weights[idx])
            history[f"WVloss_{name}"].append(valid_losses[idx] * loss_weights[idx])
        for idx, name in enumerate(eval_names):
            history[f"Teval_{name}"]["mean"].append(train_evals[idx])
            history[f"Veval_{name}"]["mean"].append(valid_evals[idx])

        if calc_stats:
            for idx, sample_loss in enumerate(sample_losses):
                train_stats = compute_dataset_stats(
                    model, train_loader.dataset, 500, device, sample_loss, loss_stats_flags[idx]
                )
                valid_stats = compute_dataset_stats(
                    model, valid_loader.dataset, 100, device, sample_loss, loss_stats_flags[idx]
                )
                if train_stats is not None:
                    for stat_key, stat_value in train_stats.items():
                        history[f"Tloss_{loss_names[idx]}"].setdefault(stat_key, []).append(stat_value)
                if valid_stats is not None:
                    for stat_key, stat_value in valid_stats.items():
                        history[f"Vloss_{loss_names[idx]}"].setdefault(stat_key, []).append(stat_value)

            for idx, sample_evaluation in enumerate(sample_evaluations):
                train_stats = compute_dataset_stats(
                    model,
                    train_loader.dataset,
                    500,
                    device,
                    sample_evaluation,
                    eval_stats_flags[idx],
                )
                valid_stats = compute_dataset_stats(
                    model,
                    valid_loader.dataset,
                    100,
                    device,
                    sample_evaluation,
                    eval_stats_flags[idx],
                )
                if train_stats is not None:
                    for stat_key, stat_value in train_stats.items():
                        history[f"Teval_{eval_names[idx]}"].setdefault(stat_key, []).append(stat_value)
                if valid_stats is not None:
                    for stat_key, stat_value in valid_stats.items():
                        history[f"Veval_{eval_names[idx]}"].setdefault(stat_key, []).append(stat_value)

        time2 = time.time() - t1
        history["epoch_time"].append(time2)

        if wandb and use_wandb:
            current_epoch_metrics = {}
            for key, value in history.items():
                if isinstance(value, list):
                    current_epoch_metrics[key] = value[-1]
                elif isinstance(value, dict):
                    for subkey, subvalues in value.items():
                        if isinstance(subvalues, list):
                            current_epoch_metrics[f"{key}.{subkey}"] = subvalues[-1]
                        else:
                            current_epoch_metrics[f"{key}.{subkey}"] = subvalues
            wandb.log(current_epoch_metrics, step=epoch)

        kit.print_epoch_results(
            epoch,
            time1,
            time2,
            loss_names,
            train_losses,
            valid_losses,
            loss_weights,
            weighted_train_loss,
            weighted_valid_loss,
            eval_names,
            train_evals,
            valid_evals,
        )

        if ckpt:
            if epoch == int(config["train"]["epochs"] * 0.25):
                torch.save(model.state_dict(), os.path.join(model_path, f"{model_prefix}_ep{epoch}.pt"))
            if epoch == int(config["train"]["epochs"] * 0.50):
                torch.save(model.state_dict(), os.path.join(model_path, f"{model_prefix}_ep{epoch}.pt"))
            if epoch == int(config["train"]["epochs"] * 0.75):
                torch.save(model.state_dict(), os.path.join(model_path, f"{model_prefix}_ep{epoch}.pt"))

    train_time = sum(history["train_time"])
    total_time = sum(history["epoch_time"])

    final_model_path = os.path.join(model_path, f"{model_prefix}.pt")
    torch.save(model.state_dict(), final_model_path)

    flat_data = []
    for idx in range(len(history["T_loss"])):
        row = {"epoch": idx + 1}
        for key, value in history.items():
            if isinstance(value, dict):
                for subkey, subvalues in value.items():
                    if isinstance(subvalues, list):
                        row[f"{key}.{subkey}"] = subvalues[idx]
                    else:
                        row[f"{key}.{subkey}"] = subvalues
            else:
                row[key] = value[idx]
        flat_data.append(row)

    columns = ["epoch"] + list(flat_data[0].keys())[1:]
    with open(os.path.join(loss_path, "history_output.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(flat_data)

    with open(os.path.join(loss_path, "train_time.txt"), "w") as file:
        file.write(str(train_time))
    with open(os.path.join(loss_path, "total_time.txt"), "w") as file:
        file.write(str(total_time))

    if wandb and use_wandb:
        wandb.log({"train_time": train_time, "total_time": total_time})
        wandb.finish()

    update_training_status(config, phase="end")
    with open(config["paths"]["config_path"], "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(f"total_time: {total_time}s")
    print("Task done!")

    return model
