# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 15:48:26 2024 (JST)

@author: Jinghao FEI
"""
import os
import yaml
import csv
import pandas as pd
import torch
from tqdm import tqdm
import time
import random
import numpy as np
from losses.LpLoss import LpLoss
from losses.InverseACLoss import InverseACLoss
from losses.RMSRE import RMSRE
from losses.R2 import R2
from losses.MSE import MSE
from utilities import utilkit as kit
from utilities.config_util import update_training_status

#%%
def compute_dataset_stats(model, dataset, batch_size, device, sample_fn, stats_flags=None):
    """
    Compute statistical metrics for a given dataset using the provided sample function.
    
    Args:
        model (torch.nn.Module): The model to generate predictions.
        dataset (torch.utils.data.Dataset): The dataset from which to compute statistics.
        batch_size (int): Batch size used for processing the dataset.
        device (torch.device): The device to perform computation.
        sample_fn (callable): Function that takes (predictions, ground_truth) as input and returns
                              a 1D torch.Tensor of metric values.
        stats_flags (list of bool or None): A list of 8 boolean values indicating whether to compute
                                            ["std", "cv", "skewness", "kurtosis", "min", 
                                            "max", "median", "variance"]. If None, all are set to False.
        
    Returns:
        dict or None: A dictionary with keys as statistical metric names and values as computed results,
                      or None if all flags are False.
    """
    # 默认不计算所有统计量（全为 False）
    if stats_flags is None:
        stats_flags = [False] * 8  # 全部不计算
    elif not any(stats_flags):
        return None  # 如果所有布尔值均为 False，则直接返回 None，跳过计算

    # 解包布尔值
    (compute_std, compute_cv, compute_skewness, compute_kurtosis,
     compute_min, compute_max, compute_median, compute_variance) = stats_flags

    model.eval()
    metric_batches = []
    with torch.no_grad():
        num_samples = len(dataset)
        for i in range(0, num_samples, batch_size):
            batch_items = [dataset[j] for j in range(i, min(i + batch_size, num_samples))]
            # Assumption: Each item in dataset is a tuple (x, f, y)
            x_batch = torch.stack([item[0] for item in batch_items]).to(device)
            f_batch = torch.stack([item[1] for item in batch_items]).to(device)
            y_batch = torch.stack([item[2] for item in batch_items]).to(device)

            # Forward pass
            outputs = model(x_batch, f_batch)
            outputs = outputs.view(outputs.size(0), -1)
            y_batch = y_batch.view(y_batch.size(0), -1)

            # Compute metric using the provided function
            metric_values = sample_fn(outputs, y_batch)
            metric_values = metric_values.unsqueeze(0)
            metric_batches.append(metric_values)

    # 合并所有 batch 的数据
    all_metrics = torch.cat(metric_batches, dim=0)

    # 统计结果字典
    stats = {}

    if compute_std:
        stats["std"] = torch.std(all_metrics, unbiased=False).item()
    if compute_variance:
        stats["variance"] = torch.var(all_metrics, unbiased=False).item()
    if compute_median:
        stats["median"] = torch.median(all_metrics).item()
    if compute_cv:
        mean_val = torch.mean(all_metrics).item()
        std_val = stats["std"] if "std" in stats else torch.std(all_metrics, unbiased=False).item()
        stats["cv"] = (std_val / mean_val) if mean_val != 0 else float('nan')
    if compute_skewness:
        mean_tensor = torch.mean(all_metrics)
        std_tensor = torch.std(all_metrics, unbiased=False)
        stats["skewness"] = torch.mean(((all_metrics - mean_tensor) / std_tensor) ** 3).item() if std_tensor != 0 else float('nan')
    if compute_kurtosis:
        mean_tensor = torch.mean(all_metrics)
        std_tensor = torch.std(all_metrics, unbiased=False)
        stats["kurtosis"] = (torch.mean(((all_metrics - mean_tensor) / std_tensor) ** 4).item() - 3) if std_tensor != 0 else float('nan')
    if compute_min:
        stats["min"] = torch.min(all_metrics).item()
    if compute_max:
        stats["max"] = torch.max(all_metrics).item()

    return stats

#%%
def train_1d(config,
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
             use_tqdm=False,
             calc_stats=True):
    update_training_status(config, phase="start")

    start_dt = datetime.now()
    timestamp = start_dt.strftime("%d%H%M")
    model_prefix = f"{config['model']['model']}-{timestamp}"

    with open(config["paths"]["config_path"], 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    loss_path = os.path.join(save_path, "loss")
    os.makedirs(loss_path, exist_ok=True)
    model_path = os.path.join(save_path, "model")
    os.makedirs(model_path, exist_ok=True)
        
    epochs = range(1, config["train"]["epochs"] + 1)
    batch_size = config["train"]["batch_size"]
    
    loss_names = [loss[0] for loss in config["loss"]["losses"]]
    loss_weights = [loss[1] for loss in config["loss"]["losses"]]
    evaluations = [evaluation[0] for evaluation in config["loss"]["evaluations"]]
    
    batch_losses = []
    batch_evaluations = []
    
    for loss in loss_names:
        new_loss = kit.load_loss_function(f"losses.{loss}.{loss}")
        loaded_loss = new_loss(size_average=True)
        batch_losses.append(loaded_loss)
        
    for evaluation in evaluations:
        new_evaluation = kit.load_loss_function(f"losses.{evaluation}.{evaluation}")
        loaded_evaluation = new_evaluation(size_average=True)
        batch_evaluations.append(loaded_evaluation)
    
    history = {}
    history["T_loss"] = []
    history["V_loss"] = []
    history["train_time"] = []
    history["epoch_time"] = []
    
    for name in loss_names:
        history[f"Tloss_{name}"] = {}
        history[f"Vloss_{name}"] = {}
        history[f"Tloss_{name}"]["mean"] = []
        history[f"Vloss_{name}"]["mean"] = []
        history[f"WTloss_{name}"] = []
        history[f"WVloss_{name}"] = []
        
    for name in evaluations:
        history[f"Teval_{name}"] = {}
        history[f"Veval_{name}"] = {}
        history[f"Teval_{name}"]["mean"] = []
        history[f"Veval_{name}"]["mean"] = []
        
    if calc_stats:
        loss_stats_flags = [loss[2] for loss in config["loss"]["losses"]]
        evaluation_stats_flags = [evaluation[1] for evaluation in config["loss"]["evaluations"]]
        
        sample_losses = []
        sample_evaluations = []
        
        for loss in loss_names:
            new_loss = kit.load_loss_function(f"losses.{loss}.{loss}")
            loaded_loss = new_loss(size_average=False)
            sample_losses.append(loaded_loss)
            
        for evaluation in evaluations:
            new_evaluation = kit.load_loss_function(f"losses.{evaluation}.{evaluation}")
            loaded_evaluation = new_evaluation(size_average=False)
            sample_evaluations.append(loaded_evaluation)
        
    if use_tqdm:
        epochs = tqdm(epochs)
        
    starttime = time.time()
    for ep in epochs:
        train_evals = np.zeros(len(batch_evaluations))
        valid_evals = np.zeros(len(batch_evaluations))
        train_losses = np.zeros(len(batch_losses))
        weighted_train_losses = np.zeros(len(batch_losses))
        valid_losses = np.zeros(len(batch_losses))
        weighted_valid_losses = np.zeros(len(batch_losses))
        
        t1 = time.time()
        
        model.train()
        for x, f, y in train_loader:
            x, f, y = x.to(device), f.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x, f)
            
            t_losses = torch.empty(len(batch_losses)).to(device)
            t_lossweights = torch.tensor(loss_weights).to(device)
            t_evaluations = torch.empty(len(batch_evaluations)).to(device)
            
            for idx, batch_evaluation in enumerate(batch_evaluations):
                t_evaluation = batch_evaluation(out.view(out.shape[0], -1), y.view(out.shape[0], -1))
                t_evaluations[idx] = t_evaluation
            for idx, batch_loss in enumerate(batch_losses):
                t_loss = batch_loss(out.view(out.shape[0], -1), y.view(out.shape[0], -1))
                t_losses[idx] = t_loss
            weighted_losses = t_losses * t_lossweights
            weighted_loss = torch.mean(weighted_losses)
            
            weighted_loss.backward()
            optimizer.step()
            
            train_losses += t_losses.detach().cpu().numpy()
            weighted_train_losses += weighted_losses.detach().cpu().numpy()
            train_evals += t_evaluations.detach().cpu().numpy()
    
        scheduler.step()
        
        time1 = time.time() - t1
        
        model.eval()
        with torch.no_grad():
            for x, f, y in valid_loader:
                x, f, y = x.to(device), f.to(device), y.to(device)
                out = model(x, f)
                
                v_losses = torch.empty(len(batch_losses)).to(device)
                v_lossweights = t_lossweights
                v_evaluations = torch.empty(len(batch_evaluations)).to(device)
                
                for idx, batch_evaluation in enumerate(batch_evaluations):
                    v_evaluation = batch_evaluation(out.view(out.shape[0], -1), y.view(out.shape[0], -1))
                    v_evaluations[idx] = v_evaluation
                for idx, batch_loss in enumerate(batch_losses):
                    v_loss = batch_loss(out.view(out.shape[0], -1), y.view(out.shape[0], -1))
                    v_losses[idx] = v_loss
                weighted_losses = v_losses * v_lossweights
                weighted_loss = torch.mean(weighted_losses)
                
                valid_losses += v_losses.detach().cpu().numpy()
                weighted_valid_losses += weighted_losses.detach().cpu().numpy()
                valid_evals += v_evaluations.detach().cpu().numpy()
        
        len_Tloader = len(train_loader)
        len_Vloader = len(valid_loader)
        train_evals /= len_Tloader
        valid_evals /= len_Vloader
        train_losses /= len_Tloader
        weighted_train_losses /= len_Tloader
        valid_losses /= len_Vloader
        weighted_valid_losses /= len_Vloader
        
        history["T_loss"].append(np.mean(weighted_train_losses))
        history["V_loss"].append(np.mean(weighted_valid_losses))
        history["train_time"].append(time1)
        
        for idx, name in enumerate(loss_names):
            history[f"Tloss_{name}"]["mean"].append(train_losses[idx])
            history[f"Vloss_{name}"]["mean"].append(valid_losses[idx])
            history[f"WTloss_{name}"].append(weighted_train_losses[idx])
            history[f"WVloss_{name}"].append(weighted_valid_losses[idx])
        for idx, name in enumerate(evaluations):
            history[f"Teval_{name}"]["mean"].append(train_evals[idx])
            history[f"Veval_{name}"]["mean"].append(valid_evals[idx])
                
        if calc_stats:
            for idx, sample_loss in enumerate(sample_losses):
                train_stats = compute_dataset_stats(model, train_loader.dataset, 500, device, sample_loss, loss_stats_flags[idx])
                valid_stats = compute_dataset_stats(model, valid_loader.dataset, 100, device, sample_loss, loss_stats_flags[idx])
                if train_stats is not None:
                    for stat_key, stat_value in train_stats.items():
                        # If key does not exist or is not a list, initialize it
                        if stat_key not in history[f"Tloss_{loss_names[idx]}"]:
                            history[f"Tloss_{loss_names[idx]}"][stat_key] = []
                        history[f"Tloss_{loss_names[idx]}"][stat_key].append(stat_value)
                if valid_stats is not None:
                    for stat_key, stat_value in valid_stats.items():
                        if stat_key not in history[f"Vloss_{loss_names[idx]}"]:
                            history[f"Vloss_{loss_names[idx]}"][stat_key] = []
                        history[f"Vloss_{loss_names[idx]}"][stat_key].append(stat_value)
            
            for idx, sample_evaluation in enumerate(sample_evaluations):
                train_stats = compute_dataset_stats(model, train_loader.dataset, 500, device, sample_evaluation, evaluation_stats_flags[idx])
                valid_stats = compute_dataset_stats(model, valid_loader.dataset, 100, device, sample_evaluation, evaluation_stats_flags[idx])
                if train_stats is not None:
                    for stat_key, stat_value in train_stats.items():
                        if stat_key not in history[f"Teval_{evaluations[idx]}"]:
                            history[f"Teval_{evaluations[idx]}"][stat_key] = []
                        history[f"Teval_{evaluations[idx]}"][stat_key].append(stat_value)
                if valid_stats is not None:
                    for stat_key, stat_value in valid_stats.items():
                        if stat_key not in history[f"Veval_{evaluations[idx]}"]:
                            history[f"Veval_{evaluations[idx]}"][stat_key] = []
                        history[f"Veval_{evaluations[idx]}"][stat_key].append(stat_value)

        time2 = time.time() - t1
        history["epoch_time"].append(time2)
        
        # Log current epoch metrics to wandb without affecting complete history
        if wandb and use_wandb:
            current_epoch_metrics = {}
            # Process top-level keys which are lists
            for key, value in history.items():
                if isinstance(value, list):
                    current_epoch_metrics[key] = value[-1]
                elif isinstance(value, dict):
                    for subkey, subvalues in value.items():
                        # If subvalues is a list, extract the latest value
                        if isinstance(subvalues, list):
                            current_epoch_metrics[f"{key}.{subkey}"] = subvalues[-1]
                        else:
                            current_epoch_metrics[f"{key}.{subkey}"] = subvalues
            wandb.log(current_epoch_metrics, step=ep)
            
        print()    
        print("=" * 50) 
        print()  # 打印一个空行
        print("Epoch Infomations")
        print(f"Epoch: {ep}")
        print(f"Training Time: {time1}")
        print(f"Epoch Time: {time2}")
        print()  # 打印一个空行
        print("Training Losses")
        for idx, name in enumerate(loss_names):
            print(f"{name}: {train_losses[idx]} | weight: {loss_weights[idx]} | weighted {name}: {weighted_train_losses[idx]}")
        print(f"Total loss: {history['T_loss'][-1]}")
        print()  # 打印一个空行
        print("Training Evaluations")
        for idx, name in enumerate(evaluations):
            print(f"{name}: {train_evals[idx]}")
        print()  # 打印一个空行
        print("Validation Losses")
        for idx, name in enumerate(loss_names):
            print(f"{name}: {valid_losses[idx]} | weight: {loss_weights[idx]} | weighted {name}: {weighted_valid_losses[idx]}")
        print(f"Total loss: {history['V_loss'][-1]}")
        print()  # 打印一个空行
        print("Validation Evaluations")
        for idx, name in enumerate(evaluations):
            print(f"{name}: {valid_evals[idx]}")
        print()
        print("=" * 50)
        print()
        
        if ckpt:
            if ep == int(config["train"]["epochs"] * 0.25):
                torch.save(model, os.path.join(model_path, config["project"]["case"] + "_25%"))
            if ep == int(config["train"]["epochs"] * 0.50):
                torch.save(model, os.path.join(model_path, config["project"]["case"] + "_50%"))
            if ep == int(config["train"]["epochs"] * 0.75):
                torch.save(model, os.path.join(model_path, config["project"]["case"] + "_75%"))
    
    train_time = sum(history["train_time"])
    total_time = sum(history["epoch_time"])
    
    torch.save(model, kit.safe_path(os.path.join(model_path, config["project"]["case"])))
        
    flat_data = []
    # Use length of one of the lists as epoch count, here history["T_loss"]记录了每个epoch的指标
    for idx in range(len(history["T_loss"])):
        row = {"epoch": idx + 1}  # 如果希望 epoch 从1开始显示
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
    
    # 获取所有列名
    columns = ["epoch"] + list(flat_data[0].keys())[1:]
    
    # 保存 CSV
    with open(kit.safe_path(os.path.join(loss_path, "history_output.csv")), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(flat_data)
    
    with open(kit.safe_path(os.path.join(loss_path, 'train_time.txt')), 'w') as file:
        file.write(str(train_time))
    with open(kit.safe_path(os.path.join(loss_path, 'total_time.txt')), 'w') as file:
        file.write(str(total_time))
    
    if wandb and use_wandb:
        wandb.log({"train_time": train_time,
                   "total_time": total_time})
        wandb.finish()
        
    update_training_status(config, phase="end")
    with open(config["paths"]["config_path"], 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"total_time: {total_time}s")
    print("Task done!")
    
    return model
