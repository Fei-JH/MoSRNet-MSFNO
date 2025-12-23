"""
Author: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
Date: 2025-08-12 18:07:20
LastEditors: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
LastEditTime: 2025-08-13 19:47:48
"""

from datetime import datetime
import os
import yaml


def is_subset_dict(small, big):
    """
    Check whether all keys and values in 'small' are contained in 'big'.
    """
    for key, value in small.items():
        if key not in big:
            return False
        if isinstance(value, dict) and isinstance(big[key], dict):
            if not is_subset_dict(value, big[key]):
                return False
        elif big[key] != value:
            return False
    return True


def check_config_file(config_dir, config, model_name, subset, gentime):
    """
    Check whether an identical config already exists in the config directory.

    Returns:
        (flag, filename):
            flag = 1 if a matching config exists, otherwise 0.
            filename is the existing or newly generated config filename.
    """
    candidate_files = []
    identical_file = None

    for filename in os.listdir(config_dir):
        if filename.endswith(".yaml") and f"{model_name}-{subset}" in filename:
            candidate_files.append(filename)
            full_path = os.path.join(config_dir, filename)
            with open(full_path, "r", encoding="utf-8") as f:
                file_config = yaml.safe_load(f)
            if is_subset_dict(config, file_config):
                identical_file = filename
                break

    if identical_file is not None:
        print(f"An identical config already existed: {identical_file}")
        return 1, identical_file

    new_n = f"{len(candidate_files) + 1:02d}"
    new_filename = f"{model_name}-{subset}-run{new_n}-{gentime}.yaml"
    return 0, new_filename


def save_config_to_yaml(config, config_path):
    """Save a configuration dictionary to a YAML file."""
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"Configuration successfully saved to {config_path}")


def update_training_status(config, phase="start"):
    """
    Update training status information in config["status"].
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if "status" not in config:
        config["status"] = {"train_count": 0, "train_history": []}

    status_info = config["status"]

    if phase == "start":
        status_info["train_count"] += 1
        status_info["train_history"].append({"start_time": now, "end_time": None})
        print(f"Training started at {now}.")
    elif phase == "end":
        if status_info["train_history"] and status_info["train_history"][-1]["end_time"] is None:
            status_info["train_history"][-1]["end_time"] = now
            print(f"Training ended at {now}.")
        else:
            print("Warning: No ongoing training session found to end.")
    else:
        print("Invalid phase specified. Use 'start' or 'end'.")

    return config
