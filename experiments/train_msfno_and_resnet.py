'''
Author: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
Date: 2025-08-12 18:06:31
LastEditors: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
LastEditTime: 2025-08-14 16:49:46
'''


import os
import yaml
import csv
import torch
from tqdm import tqdm
import time
import numpy as np
from datetime import datetime
from utilities import utilkit as kit
from utilities.config_util import update_training_status
from utilities.train_util import compute_dataset_stats


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
             use_tqdm=True,
             calc_stats=True):
    
    update_training_status(config, phase="start")

    start_dt = datetime.now()
    timestamp = start_dt.strftime("%y%m%d%H%M%S")
    model_prefix = f"{config['model']['model']}-{timestamp}"

    with open(config["paths"]["config_path"], 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    loss_path = os.path.join(save_path, "loss")
    os.makedirs(loss_path, exist_ok=True)
    model_path = os.path.join(save_path, "model")
    os.makedirs(model_path, exist_ok=True)
        
    epochs = range(1, config["train"]["epochs"] + 1)
    
    loss_names = [loss[0] for loss in config["loss"]["losses"]]
    loss_weights = [loss[1] for loss in config["loss"]["losses"]]
    evaluations = [evaluation[0] for evaluation in config["loss"]["evaluations"]]
    
    batch_losses = []
    batch_evaluations = []
    
    for loss in config["loss"]["losses"]:
        new_loss = kit.load_loss_function(f"losses.{loss[0].lower()}.{loss[0]}")
        loaded_loss = new_loss(**loss[-1])
        batch_losses.append(loaded_loss)
        
    for evaluation in config["loss"]["evaluations"]:
        new_evaluation = kit.load_loss_function(f"losses.{evaluation[0].lower()}.{evaluation[0]}")
        loaded_evaluation = new_evaluation(**evaluation[-1])
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
            loaded_loss = new_loss(size_average=False, reduction=False)
            sample_losses.append(loaded_loss)
        for evaluation in evaluations:
            new_evaluation = kit.load_loss_function(f"losses.{evaluation}.{evaluation}")
            loaded_evaluation = new_evaluation(size_average=False, reduction=False)
            sample_evaluations.append(loaded_evaluation)

    starttime = time.time()
    for ep in epochs:
        train_evals = np.zeros(len(batch_evaluations))
        valid_evals = np.zeros(len(batch_evaluations))
        train_losses = np.zeros(len(batch_losses))
        weighted_train_loss = np.zeros(1)
        valid_losses = np.zeros(len(batch_losses))
        weighted_valid_loss = np.zeros(1)

        ep_str = f"[ EPOCH {ep:03d} ]"
        total_len = 80
        side_len = (total_len - len(ep_str)) // 2
        extra = (total_len - len(ep_str)) % 2
        print("=" * side_len + ep_str + "=" * (side_len + extra))

        t1 = time.time()
        model.train()
        train_iter = train_loader
        if use_tqdm:
            bar_format = '{desc}|{bar}| Speed:{rate_fmt} {postfix}'
            train_iter = tqdm(
                train_loader,
                desc=f"[Epoch {ep}] Training ",
                leave=True,
                ncols=80,
                bar_format=bar_format
            )
            total_batch_time = 0.0
        for mode, dmg in train_iter:
            if use_tqdm:
                train_t = time.time()
            mode, dmg = mode.to(device), dmg.to(device)
            
            optimizer.zero_grad()
            out = model(mode)
                                    
            t_losses = torch.empty(len(batch_losses)).to(device)
            lossweights = torch.tensor(loss_weights).to(device)
            t_evaluations = torch.empty(len(batch_evaluations)).to(device)
            
            for idx, batch_evaluation in enumerate(batch_evaluations):
                evaluation_name = batch_evaluation.__class__.__name__
                t_evaluation = batch_evaluation(out.view(out.shape[0], -1), dmg.view(dmg.shape[0], -1))
                t_evaluations[idx] = t_evaluation
            for idx, batch_loss in enumerate(batch_losses):
                loss_name = batch_loss.__class__.__name__
                t_loss = batch_loss(out.view(out.shape[0], -1), dmg.view(dmg.shape[0], -1))
                t_losses[idx] = t_loss

            weighted_t_losses = t_losses * lossweights
            weighted_t_loss = torch.sum(weighted_t_losses)

            weighted_t_loss.backward()
            optimizer.step()
            train_losses += t_losses.detach().cpu().numpy()
            weighted_train_loss += weighted_t_loss.detach().cpu().numpy()
            train_evals += t_evaluations.detach().cpu().numpy()
            
            if use_tqdm:
                batch_time = time.time() - train_t
                total_batch_time += batch_time
                total_str = f"{total_batch_time:7.3f} s"
                train_iter.set_postfix_str(f'Total:{total_str}')
                
        scheduler.step()
        time1 = time.time() - t1
        model.eval()
        with torch.no_grad():
            valid_iter = valid_loader
            if use_tqdm:
                bar_format = '{desc}|{bar}| Speed:{rate_fmt} {postfix}'
                valid_iter = tqdm(
                    valid_loader,
                    desc=f"[Epoch {ep}] Validation",
                    leave=True,
                    ncols=80,
                    bar_format=bar_format
                )
                total_valid_time = 0.0
            for mode, dmg in valid_iter:
                if use_tqdm:
                    valid_t = time.time()
                mode, dmg = mode.to(device, non_blocking=True), dmg.to(device, non_blocking=True)
                out = model(mode)
                
                v_losses = torch.empty(len(batch_losses)).to(device)
                v_evaluations = torch.empty(len(batch_evaluations)).to(device)
                
                for idx, batch_evaluation in enumerate(batch_evaluations):
                    evaluation_name = batch_evaluation.__class__.__name__
                    v_evaluation = batch_evaluation(out.view(out.shape[0], -1), dmg.view(dmg.shape[0], -1))
                    v_evaluations[idx] = v_evaluation
                for idx, batch_loss in enumerate(batch_losses):
                    loss_name = batch_loss.__class__.__name__
                    v_loss = batch_loss(out.view(out.shape[0], -1), dmg.view(dmg.shape[0], -1))
                    v_losses[idx] = v_loss

                weighted_v_losses = v_losses * lossweights
                weighted_v_loss = torch.sum(weighted_v_losses)

                valid_losses += v_losses.detach().cpu().numpy()
                weighted_valid_loss += weighted_v_loss.detach().cpu().numpy()
                valid_evals += v_evaluations.detach().cpu().numpy()
     
                if use_tqdm:
                    batch_time = time.time() - valid_t
                    total_valid_time += batch_time
                    total_str = f"{total_valid_time:7.3f} s"
                    valid_iter.set_postfix_str(f'Total:{total_str}')

        len_Tloader = len(train_loader)
        len_Vloader = len(valid_loader)
        train_evals /= len_Tloader
        valid_evals /= len_Vloader
        train_losses /= len_Tloader
        weighted_train_loss /= len_Tloader
        valid_losses /= len_Vloader
        weighted_valid_loss /= len_Vloader

        history["T_loss"].append(weighted_train_loss.item())
        history["V_loss"].append(weighted_valid_loss.item())
        history["train_time"].append(time1)

        for idx, name in enumerate(loss_names):
            history[f"Tloss_{name}"]["mean"].append(train_losses[idx])
            history[f"Vloss_{name}"]["mean"].append(valid_losses[idx])
            history[f"WTloss_{name}"].append(train_losses[idx]*loss_weights[idx])
            history[f"WVloss_{name}"].append(valid_losses[idx]*loss_weights[idx])
        for idx, name in enumerate(evaluations):
            history[f"Teval_{name}"]["mean"].append(train_evals[idx])
            history[f"Veval_{name}"]["mean"].append(valid_evals[idx])
                
        if calc_stats:
            for idx, sample_loss in enumerate(sample_losses):
                train_stats = compute_dataset_stats(model, train_loader.dataset, 500, device, sample_loss, loss_stats_flags[idx])
                valid_stats = compute_dataset_stats(model, valid_loader.dataset, 100, device, sample_loss, loss_stats_flags[idx])
                if train_stats is not None:
                    for stat_key, stat_value in train_stats.items():
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
            for key, value in history.items():
                if isinstance(value, list):
                    current_epoch_metrics[key] = value[-1]
                elif isinstance(value, dict):
                    for subkey, subvalues in value.items():
                        if isinstance(subvalues, list):
                            current_epoch_metrics[f"{key}.{subkey}"] = subvalues[-1]
                        else:
                            current_epoch_metrics[f"{key}.{subkey}"] = subvalues
            wandb.log(current_epoch_metrics, step=ep)
            
        kit.print_epoch_results(
                            ep, time1, time2,
                            loss_names, train_losses, valid_losses, loss_weights, weighted_train_loss, weighted_valid_loss,
                            evaluations, train_evals, valid_evals
                            )
        
        if ckpt:
            if ep == int(config["train"]["epochs"] * 0.25):
                torch.save(model.state_dict(), os.path.join(model_path, f"{model_prefix}_ep{ep}.pt"))
            if ep == int(config["train"]["epochs"] * 0.50):
                torch.save(model.state_dict(), os.path.join(model_path, f"{model_prefix}_ep{ep}.pt"))
            if ep == int(config["train"]["epochs"] * 0.75):
                torch.save(model.state_dict(), os.path.join(model_path, f"{model_prefix}_ep{ep}.pt"))

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
    
    with open(os.path.join(loss_path, 'train_time.txt'), 'w') as file:
        file.write(str(train_time))
    with open(os.path.join(loss_path, 'total_time.txt'), 'w') as file:
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