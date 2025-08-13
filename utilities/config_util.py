# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:28:00 2025 (JST)

@author: Jinghao FEI
"""
import os
import yaml
#%%


def is_subset_dict(small, big):
    """
    Check whether all keys and values in dictionary 'small' are contained in dictionary 'big'.
    For nested dictionaries, the check is performed recursively.
    """
    for key, value in small.items():
        if key not in big:
            return False
        if isinstance(value, dict) and isinstance(big[key], dict):
            if not is_subset_dict(value, big[key]):
                return False
        else:
            if big[key] != value:
                return False
    return True


def check_config_file(config_dir, config, model_name, subset, tasktype, gentime):
    """
    This function searches for YAML configuration files in the specified directory
    whose filenames contain the substring f"{model_name}-{subset}-{tasktype}".
    The filename format is assumed as:
        EXP{n}-{model_name}-{subset}-{tasktype}-{gentime}.yaml

    If a file is found such that the given 'config' is a subset of the file's configuration,
    the function returns (1, file_relative_path).
    Otherwise, it counts the number of candidate files and assigns n = (candidate count) + 1,
    constructs a new filename, and returns (0, new_file_relative_path).

    Parameters:
        config_dir (str): Directory containing configuration YAML files.
        config (dict): The current configuration dictionary.
        model_name (str): Model name.
        subset (str): Subset identifier.
        tasktype (str): Task type.
        gentime (str): Generated timestamp string.
    
    Returns:
        tuple: (flag, relative_path)
            flag: 1 if an identical configuration file (i.e., file's config contains all items of current config) exists,
                  0 otherwise.
            relative_path: Relative path of the identical or newly constructed config file.
    """
    candidate_files = []
    identical_file = None

    # 遍历配置目录下所有 YAML 文件
    for fname in os.listdir(config_dir):
        if fname.endswith('.yaml') and f"{model_name}-{subset}-{tasktype}" in fname:
            candidate_files.append(fname)
            full_path = os.path.join(config_dir, fname)
            with open(full_path, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f)
            # 只检查代码中已有的项是否全部重合（即 config 是 file_config 的子集）
            if is_subset_dict(config, file_config):
                identical_file = fname
                break

    if identical_file is not None:
        print(f"An identical config already existed：{identical_file}")
        return 1, identical_file
    else:
        # n 的值为满足条件的文件数 + 1
        new_n = len(candidate_files) + 1
        new_fname = f"EXP{new_n}-{model_name}-{subset}-{tasktype}-{gentime}.yaml"
        return 0,  new_fname


def save_config_to_yaml(config, config_path):
    """
    Save the configuration dictionary to a YAML file at the specified path.
    """
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"Configuration successfully saved to {config_path}") 
    
    
def update_training_status(config, phase="start"):
    """
    更新 config["status"] 中的训练状态信息，包括训练次数和时间戳记录。
    
    Parameters:
        config (dict): 配置字典，需包含或初始化 "status" 键。
        phase (str): 状态更新类型，"start" 表示训练开始，"end" 表示训练结束。
    
    Returns:
        dict: 更新后的配置字典。
    """
    from datetime import datetime
    # Get current timestamp
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Initialize the status field if not present
    if "status" not in config:
        config["status"] = {
            "train_count": 0,
            "train_history": []
        }
    
    # Access the status dictionary
    status_info = config["status"]
    
    if phase == "start":
        # Increment training count and add a new training record with start_time
        status_info["train_count"] += 1
        status_info["train_history"].append({
            "start_time": now,  # Record the training start time
            "end_time": None    # End time is unknown at training start
        })
        print(f"Training started at {now}.")
        
    elif phase == "end":
        # Update the last training record with end_time if it exists and is not finished
        if status_info["train_history"] and status_info["train_history"][-1]["end_time"] is None:
            status_info["train_history"][-1]["end_time"] = now
            print(f"Training ended at {now}.")
        else:
            print("Warning: No ongoing training session found to end.")
    else:
        print("Invalid phase specified. Use 'start' or 'end'.")
    
    return config