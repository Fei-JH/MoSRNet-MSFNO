'''
Author: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
Date: 2025-08-12 18:07:20
LastEditors: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
LastEditTime: 2025-08-13 18:06:46
'''


import os
import yaml


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


def check_config_file(config_dir, config, model_name, subset, gentime):
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
        if fname.endswith('.yaml') and f"{model_name}-{subset}" in fname:
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
        new_n = f"{len(candidate_files) + 1:02d}"
        new_fname = f"{model_name}-{subset}-EXP{new_n}-{gentime}.yaml"
        return 0,  new_fname


def save_config_to_yaml(config, config_path):
    """
    Save the configuration dictionary to a YAML file at the specified path.
    """
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"Configuration successfully saved to {config_path}") 
    
