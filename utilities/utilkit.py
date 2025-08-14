# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:42:26 2024

@author:FEI JINGHAO
"""
import os
import numpy as np
import pandas as pd
import shutil
import importlib
import sys
import socket

from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d

import platform
import subprocess
import psutil

from datetime import datetime

import torch
import random


def create_folder_batches(workdir, foldername):
    """
    Create a folder with a timestamp and a given name in the specified working directory.
    If the folder already exists, a suffix (1), (2), etc., is added to create a unique folder name.

    Parameters:
    - workdir: The base directory where the folder will be created.
    - foldername: The base name of the folder to be created, which will be prefixed with a timestamp.

    Returns:
    - The path of the newly created folder.
    """
    # Get current date and time
    current_datetime = datetime.now()
    
    # Format the timestamp
    timestamp = current_datetime.strftime("%Y%m%d-%H%M")
    
    # Combine timestamp and foldername
    full_folder_name = f"{timestamp} {foldername}"
    
    # Initial folder path
    new_folder_path = os.path.join(workdir, full_folder_name)
    suffix = 1

    # Check if the folder exists and create a unique folder name if necessary
    while os.path.exists(new_folder_path):
        new_folder_path = os.path.join(workdir, f"{full_folder_name}({suffix})")
        suffix += 1

    # Create the folder with the final unique path
    os.makedirs(new_folder_path, exist_ok=True)

    return new_folder_path

def create_folder(workdir, foldername):
    """
    Create a folder with a timestamp and a given name in the specified working directory.
    If the folder already exists, a suffix (1), (2), etc., is added to create a unique folder name.

    Parameters:
    - workdir: The base directory where the folder will be created.
    - foldername: The base name of the folder to be created, which will be prefixed with a timestamp.

    Returns:
    - The path of the newly created folder.
    """    
    
    # Initial folder path
    new_folder_path = os.path.join(workdir, foldername)

    # Check if the folder exists and create a unique folder name if necessary
    if os.path.exists(new_folder_path):
        print(f"{foldername} already exists")
        pass
    else:
        # Create the folder with the final unique path
        os.makedirs(new_folder_path)

    return new_folder_path


def get_cpu_name_windows():
    try:
        # 执行PowerShell命令获取CPU信息
        result = subprocess.run(["powershell", "-Command", "Get-WmiObject Win32_Processor | Format-Table Name -HideTableHeaders"], capture_output=True, text=True)
        # 尝试直接提取CPU名称，忽略空行和标题
        lines = result.stdout.strip().splitlines()
        cpu_name = lines[0].strip() if lines else "Unknown CPU"
        return cpu_name
    except Exception as e:
        print(f"Error fetching CPU name: {e}")
        return "Error"


def get_cpu_info():
    cpu_logical_cores = psutil.cpu_count()
    cpu_physical_cores = psutil.cpu_count(logical=False)
    cpu_usage_percent = psutil.cpu_percent(interval=1)
    cpu_name = get_cpu_name_windows()
    return {
        "cpu_name": cpu_name,
        "cpu_logical_cores": cpu_logical_cores,
        "cpu_physical_cores": cpu_physical_cores,
        "cpu_usage_percent": cpu_usage_percent
    }


def get_gpu_info():
    try:
        nvidia_smi_output = subprocess.check_output(['nvidia-smi', '--query-gpu=gpu_name,memory.total', '--format=csv,noheader,nounits']).decode('utf-8')
        gpu_name, total_memory = nvidia_smi_output.strip().split('\n')[0].split(', ')
        return {"gpu_name": gpu_name, "total_memory_gb": int(total_memory)/1024}
    except Exception as e:
        print(f"Error fetching GPU information: {e}")
        return {"gpu_name": None, "total_memory_gb": 0}


def get_ram_info():
    ram = psutil.virtual_memory()
    total_ram_gb = ram.total / (1024 ** 3)
    available_ram_gb = ram.available / (1024 ** 3)
    return {"total_ram_gb": total_ram_gb, "available_ram_gb": available_ram_gb}


def load_training_data(in_chan=5, out_chan=1, dataset_dir=r".\TRAINING DATASET",freq=False):
    x_data = {}
    y_data = {}

    # 加载输入数据

    a = list(range(1, in_chan + 1))
        
    for i in a:
        file_path = f"{dataset_dir}\\L_mode{i}.csv"
        df = pd.read_csv(file_path, header=None)
        x_data[f'x{i}_data'] = torch.tensor(df.values, dtype=torch.float32)
        
    
    # 加载输出数据
    y_data_path = f"{dataset_dir}\\L_DMG.csv"
    df_y = pd.read_csv(y_data_path, header=None)
    y_data['y1_data'] = torch.tensor(df_y.values, dtype=torch.float32)
    if out_chan == 2:
        y_data_path_R = f"{dataset_dir}\\R_DMG.csv"
        df_y_R = pd.read_csv(y_data_path_R, header=None)
        y_data['y2_data'] = torch.tensor(df_y_R.values, dtype=torch.float32)
        
    # 加载频率数据
    if freq:
        file_path_freq = f"{dataset_dir}\\FREQ.csv"
        df_freq = pd.read_csv(file_path_freq, header=None)
        freq_data = torch.tensor(df_freq.values, dtype=torch.float32)[:,:in_chan]
    else:
        freq_data = None

    return x_data, y_data,freq_data


def load_test_data(in_chan=5, side=1, freq=False, dataset_dir=r".\TRAINING DATASET"):
    x_data = {}

    # 加载输入数据
    if in_chan ==2:
        a = [1, 3]
    # if in_chan ==4:
    #     a = [1, 3, 4, 5]
    else:
        a = list(range(1, in_chan + 1))
        
    for i in a:
        if side == 1:
            file_path = f"{dataset_dir}\\L_mode{i}.csv"
            df = pd.read_csv(file_path, header=None)
            x_data[f'x{i}_data'] = torch.tensor(df.values, dtype=torch.float32)
        elif side == 2:
            for prefix in ('L', 'R'):
                file_path = f"{dataset_dir}\\{prefix}_mode{i}.csv"
                df = pd.read_csv(file_path, header=None)
                x_data[f'x{i}{1 if prefix == "L" else 2}_data'] = torch.tensor(df.values, dtype=torch.float32)
    
    # # 加载频率数据
    # if freq:
    #     for i in range(1, in_chan + 1):
    #         file_path_freq = f"{dataset_dir}\\freq{i}.csv"
    #         df_freq = pd.read_csv(file_path_freq, header=None)
    #         x_data[f'x{i}3_data'] = torch.tensor(df_freq.values, dtype=torch.float32)
    
    return x_data


def nearest_neighbor(data, n_output):
    """
    Nearest neighbor resampling function that adjusts the data from its original length to the target length.
    
    :param data: The input data sequence, a PyTorch tensor.
    :param n_output: The target sequence length.
    :return: The adjusted data sequence.
    """
    n_input = data.shape[1]
    indices = torch.linspace(0, n_input - 1, n_output).long()
    new_data = torch.index_select(data, 1, indices)
    return new_data

def linear(data, n_output):
    """
    Linear interpolation for batched data in PyTorch tensors.
    
    :param data: Input data as a PyTorch tensor with shape [batch_size, seq_len, ...].
    :param n_output: Target sequence length after interpolation.
    :return: Interpolated data with shape [batch_size, n_output, ...].
    """
    batch_size, n_input, *rest_dims = data.shape
    if n_output == 1:
        return data[:, -1:, ...]  # Keep the last item along the sequence dimension

    # Calculate scale for linear interpolation
    scale_factor = (n_input - 1) / (n_output - 1)
    new_indices = torch.linspace(0, n_input - 1, n_output, device=data.device)
    lower_indices = new_indices.floor().long()
    upper_indices = torch.clamp(lower_indices + 1, max=n_input - 1)
    weight_upper = (new_indices - lower_indices.float()).to(data.dtype)
    weight_lower = 1.0 - weight_upper

    # Handle broadcasting for batched operations
    lower_data = torch.gather(data, 1, lower_indices.repeat(batch_size, 1, *([1]*len(rest_dims))).long())
    upper_data = torch.gather(data, 1, upper_indices.repeat(batch_size, 1, *([1]*len(rest_dims))).long())

    interpolated_data = weight_lower.view(1, -1, *([1]*len(rest_dims))) * lower_data + \
                         weight_upper.view(1, -1, *([1]*len(rest_dims))) * upper_data

    return interpolated_data


def linear_interpolation(data, x_old, x_new, nmlz=True):
    """
    Linear interpolation to adjust the sequence length for a batch of sequences.
    
    :param data: The input data sequences, a 2D PyTorch tensor [batch_size, seq_len].
    :param x_old: The original sequence indices as a numpy array or a list.
    :param x_new: The target sequence indices as a numpy array or a list.
    :param nmlz: Normalization flag to normalize the interpolated data.
    :return: The adjusted data sequences as a 2D PyTorch tensor [batch_size, new_seq_len].
    """
    # Initialize a list to hold interpolated sequences
    interpolated_seqs = []
    
    # Ensure x_old and x_new are numpy arrays for consistency
    x_old = np.asarray(x_old)
    x_new = np.asarray(x_new)
    
    # Loop over each sequence in the batch
    for i in range(data.shape[0]):
        # Convert current sequence to numpy for processing
        seq_np = data[i].cpu().numpy()  # Ensure tensor is on CPU for conversion
        
        # Create a linear interpolator
        linear_interp = interp1d(x_old, seq_np, kind='linear', fill_value="extrapolate")
        
        # Interpolate
        new_seq_np = linear_interp(x_new)
        
        # if nmlz:
        #     # Normalize the data if the normalization flag is True
        #     new_seq_np /= np.max(np.abs(new_seq_np)) if np.max(np.abs(new_seq_np)) != 0 else 1
        
        # Convert back to PyTorch tensor and add to the list
        interpolated_seqs.append(torch.from_numpy(new_seq_np).to(data.device))
    
    # Stack all interpolated sequences to form a new tensor
    new_data = torch.stack(interpolated_seqs, dim=0)
    new_data = new_data.to(dtype=torch.float)
    
    return new_data


def cubic_spline(data, x_old, x_new, nmlz=True):
    """
    Cubic spline interpolation to adjust the sequence length for a batch of sequences.
    
    :param data: The input data sequences, a 2D PyTorch tensor [batch_size, seq_len].
    :param x_old: The original sequence indices as a numpy array or a list.
    :param x_new: The target sequence indices as a numpy array or a list.
    :return: The adjusted data sequences as a 2D PyTorch tensor [batch_size, new_seq_len].
    """
    # Initialize a list to hold interpolated sequences
    interpolated_seqs = []
    
    # Ensure x_old and x_new are numpy arrays for consistency
    x_old = np.asarray(x_old)
    x_new = np.asarray(x_new)
    
    # Loop over each sequence in the batch
    for i in range(data.shape[0]):
        # Convert current sequence to numpy for processing
        seq_np = data[i].cpu().numpy()  # Ensure tensor is on CPU for conversion
        
        # Create a cubic spline interpolator
        cs = CubicSpline(x_old, seq_np, bc_type='natural')
        
        # Interpolate
        new_seq_np = cs(x_new)
        
        if nmlz:
            pass
            # new_seq_np /= max(abs(new_seq_np))
        
        # Convert back to PyTorch tensor and add to the list
        interpolated_seqs.append(torch.from_numpy(new_seq_np).to(data.device))
    
    # Stack all interpolated sequences to form a new tensor
    new_data = torch.stack(interpolated_seqs, dim=0)
    new_data = new_data.to(dtype=torch.float)
    
    return new_data 

def replace_out_of_bounds_y(data, x_new, x_old_range, fixed_value):
    """
    Replace values in the y-axis data that correspond to x-axis points outside the original x-axis data range
    with a fixed value.
    
    :param data: The y-axis data as a PyTorch tensor, after interpolation.
    :param x_new: The x-axis values after interpolation as a numpy array or a PyTorch tensor.
    :param x_old_range: A tuple or list with two elements, indicating the original x-axis data range (min, max).
    :param fixed_value: The value to replace the out-of-bounds y-axis data with.
    :return: A PyTorch tensor with out-of-bounds y-axis values replaced.
    """
    # Ensure data is a PyTorch tensor
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)
    if not isinstance(x_new, torch.Tensor):
        x_new = torch.tensor(x_new, dtype=torch.float32)
    
    # Extract the original min and max x-axis range values
    x_old_min, x_old_max = x_old_range
    
    # Create a mask for values outside the x_old range
    out_of_bounds_mask = (x_new < x_old_min) | (x_new > x_old_max)
    
    # Replace out-of-bounds values with the fixed value
    data[out_of_bounds_mask] = fixed_value
    
    return data

def interpolate(x_data, y_data, target_length=None, interpolator=nearest_neighbor):
    """
    Adjusts and interpolates all sequences in x_data and y_data, allowing the specification of different interpolation methods.

    :param x_data: The input data dictionary.
    :param y_data: The output data dictionary.
    :param target_length: The target interpolation length. If not specified, the maximum length among the sequences is used.
    :param interpolator: The function to be used for interpolation.
    """
    # Determine the max length among all sequences
    max_length = max([data.shape[1] for key, data in x_data.items()] + 
                     [data.shape[1] for key, data in y_data.items()])

    # Determine the target length for interpolation
    if target_length is None:
        target_length = max_length
    elif target_length < max_length:
        raise ValueError("The specified target length is smaller than the maximum length among the sequences.")

    # Perform interpolation on each sequence in x_data and y_data using efficient batch operations
    for key in x_data:
        if x_data[key].shape[1] < target_length:
            x_data[key] = interpolator(x_data[key], target_length)

    for key in y_data:
        if y_data[key].shape[1] < target_length:
            y_data[key] = interpolator(y_data[key], target_length)

    return x_data, y_data

def merge_tensors(data_dict):
    """
    Merges tensors from a dictionary into a single tensor.

    :param data_dict: Dictionary containing the data to merge. Each key-value pair in the dictionary
                      is assumed to be a string-tensor pair, where each tensor has the same first dimension.
    :return: A single tensor where the tensors have been concatenated along the second dimension.
    """
    # 使用字典的第一个元素初始化列表，假设所有张量的第一维大小相同
    tensors_list = [data.reshape(data.shape[0], -1, 1) for data in data_dict.values()]
    
    # 沿特定维度（第二维）合并张量
    merged_tensor = torch.cat(tensors_list, dim=2)
    
    return merged_tensor


def copy_files(src_dir, dst_dir):
    """
    复制一个目录下的所有文件到另一个目录。

    参数:
    src_dir (str): 源目录路径。
    dst_dir (str): 目标目录路径。
    """
    # 检查源目录是否存在
    if not os.path.exists(src_dir):
        print("源目录不存在，请检查路径。")
        return

    # 如果目标目录不存在，创建它
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # 遍历源目录中的文件和子目录
    for item in os.listdir(src_dir):
        src_path = os.path.join(src_dir, item)
        dst_path = os.path.join(dst_dir, item)

        # 如果是文件夹，则递归调用本函数
        if os.path.isdir(src_path):
            copy_files(src_path, dst_path)
        else:
            # 如果是文件，则复制文件
            shutil.copy2(src_path, dst_path)

def set_seed(seed):
    """Set the seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def display_system_info():
    """
    Collects and prints the system and hardware specifications.
    """
    system_info = {
        'platform': platform.system(),
        'platform_release': platform.release(),
        'platform_version': platform.version(),
        'architecture': platform.machine(),
        'cpu': platform.processor(),
        'ram': f"{round(psutil.virtual_memory().total / (1024.0 ** 3))} GB"
    }

    if torch.cuda.is_available():
        system_info['gpu'] = torch.cuda.get_device_name(0)
        system_info['gpu_count'] = torch.cuda.device_count()
    else:
        system_info['gpu'] = None
        system_info['gpu_count'] = 0

    # Print system and hardware information to the console
    print("### System and Hardware Specifications ###")
    print(f"Operating System: {system_info['platform']} {system_info['platform_release']} (Version: {system_info['platform_version']})")
    print(f"Architecture: {system_info['architecture']}")
    print(f"CPU: {system_info['cpu']}")
    print(f"Memory: {system_info['ram']}")
    if 'gpu' in system_info:
        print(f"GPU: {system_info['gpu']}")
        print(f"Number of GPUs: {system_info['gpu_count']}")
    else:
        print("GPU: No GPU available")
    print("########################################")
    
    return system_info

def loss_import(module_name, class_name):
    """
    Dynamically import a specific class from a module.
    
    Args:
        module_name (str): The name of the module (e.g., "R2").
        class_name (str): The specific class to import (e.g., "R2_Advanced").
    
    Returns:
        class: The imported class.
    """
    # Construct the module path
    module_path = f"losses.{module_name}"
    
    try:
        # Dynamically import the module
        module = importlib.import_module(module_path)
        
        # Retrieve the specific class from the module
        return getattr(module, class_name)
    except ImportError:
        raise ImportError(f"Module 'losses.{module_name}' not found.")
    except AttributeError:
        raise AttributeError(f"Class '{class_name}' not found in module 'losses.{module_name}'.")
        


def load_loss_function(module_path: str):
    """
    Dynamically imports a loss function or class from a given module.

    Parameters:
    - module_path (str): The full module path including the function/class name (e.g., 'losses.MAE.MAE').

    Returns:
    - Callable: The imported function or class.

    Raises:
    - ImportError: If the module cannot be found.
    - AttributeError: If the function/class does not exist in the module.
    - TypeError: If the retrieved object is not callable.
    """
    try:
        # Extract module name and function/class name
        *module_parts, function_name = module_path.split('.')
        module_parts[1] = module_parts[1].lower()
        module_name = ".".join(module_parts)

        # Dynamically import the module
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        raise ImportError(f"Module '{module_name}' not found. Please check if the module exists.")

    try:
        # Retrieve the function/class with the given name
        func = getattr(module, function_name)
    except AttributeError:
        raise AttributeError(f"Module '{module_name}' does not contain an attribute '{function_name}'. "
                             f"Please check if it is correctly defined in '{module_name}.py'.")

    # Ensure the function is callable
    if not callable(func):
        raise TypeError(f"The attribute '{function_name}' in module '{module_name}' is not callable. "
                        "Please ensure it is a function or a class.")

    return func  # Return the callable function/class

def safe_path(path: str) -> str:
    """
    Convert a relative or absolute path to an absolute Windows-compatible path.
    If the path exceeds 260 characters on Windows, add the '\\\\?\\' prefix.

    Parameters:
        path (str): Relative or absolute file/folder path

    Returns:
        str: Safe absolute path (with '\\\\?\\' prefix if needed on Windows)
    """
    abs_path = os.path.abspath(path)

    # if os.name == 'nt' and len(abs_path) > 260 and not abs_path.startswith(r"\\?\"):
    abs_path = r"\\?\{}".format(abs_path)

    return abs_path


def adaptive_average_downsample(x, n):
    # x: (B, C, L)
    B, C, L = x.shape
    device = x.device
    # 得到每段的起止索引
    edges = torch.linspace(0, L, steps=n+1, device=device).round().long()
    out = []
    for i in range(n):
        start, end = edges[i].item(), edges[i+1].item()
        if end > start:
            seg = x[..., start:end]  # (B, C, width)
            avg = seg.mean(dim=-1, keepdim=True)  # (B, C, 1)
        else:  # 极端情况下start==end，直接取start点
            avg = x[..., start:start+1]
        out.append(avg)
    out = torch.cat(out, dim=-1)  # (B, C, n)
    return out

def cubic_spline_interp(x_sub: torch.Tensor, src_idx: np.ndarray, tgt_idx: np.ndarray) -> torch.Tensor:
    """
    Cubic spline interpolation from subsampled sequence to target indices.
    Args:
        x_sub: Tensor, shape [N, K, C_sub], subsampled sequence (C_sub = len(src_idx))
        src_idx: 1D numpy array, source indices, length C_sub
        tgt_idx: 1D numpy array, target indices, length M
    Returns:
        Tensor, shape [N, K, M]
    """
    with torch.no_grad():
        x_sub_np = x_sub.cpu().numpy()
        N, K, C_sub = x_sub_np.shape
        M = len(tgt_idx)
        out_np = np.empty((N, K, M), dtype=np.float32)
        for n in range(N):           # batch
            for k in range(K):       # channel
                # For sample n, channel k, input is shape [C_sub]
                cs = CubicSpline(src_idx, x_sub_np[n, k, :], bc_type='natural')
                out_np[n, k, :] = cs(tgt_idx)   # [M]
        return torch.from_numpy(out_np)
    

def assemble_condensed_matrices(
    dmgfields: torch.Tensor,
    L: float,
    E: float,
    I: float,
    rho: float,
    A: float,
    n_elements: int,
    mass_dmg_power: float = 2.0,
    device: torch.device | None = None,
):
    """
    Assemble condensed mass and stiffness matrices for a 2-DOF/-node Euler beam.
    Only translational DOFs are retained via static condensation.

    Parameters
    ----------
    dmgfields : Tensor, shape [samples, 1, n_elements] or [samples, n_elements]
        Element-wise damage coefficients (>0). 1 = intact.
    L, E, I, rho, A : float
        Beam geometry & material constants.
    n_elements : int
        Number of finite elements.
    mass_dmg_power : float, optional
        Mass scaling exponent (m_coeff = dmg ** (-mass_dmg_power)).
    device : torch.device, optional
        Target device. Defaults to dmgfields.device.

    Returns
    -------
    M_cond : Tensor, shape [samples, n_nodes, n_nodes]
    K_cond : Tensor, shape [samples, n_nodes, n_nodes]
        Condensed matrices defined only on translational DOFs.
    """
    # ------------------------------------------------------------------ #
    # ---------- basic sanity & reshape -------------------------------- #
    # ------------------------------------------------------------------ #
    if dmgfields.dim() == 3:
        dmgfields = dmgfields.squeeze(1)          # [samples, n_elements]
    if dmgfields.dim() != 2 or dmgfields.size(1) != n_elements:
        raise ValueError("dmgfields must be [samples, n_elements]")

    device = device or dmgfields.device
    dtype = dmgfields.dtype
    samples = dmgfields.size(0)
    n_nodes = n_elements + 1
    dof_total = 2 * n_nodes                     # 2 DOF/node: w, θ

    # ------------------------------------------------------------------ #
    # ---------- element matrices (4×4) -------------------------------- #
    # ------------------------------------------------------------------ #
    Le = L / n_elements
    Ke_base = (E * I / Le**3) * torch.tensor(
        [
            [12, 6 * Le, -12, 6 * Le],
            [6 * Le, 4 * Le**2, -6 * Le, 2 * Le**2],
            [-12, -6 * Le, 12, -6 * Le],
            [6 * Le, 2 * Le**2, -6 * Le, 4 * Le**2],
        ],
        dtype=dtype,
        device=device,
    )
    Me_base = (rho * A * Le / 420) * torch.tensor(
        [
            [156, 22 * Le, 54, -13 * Le],
            [22 * Le, 4 * Le**2, 13 * Le, -3 * Le**2],
            [54, 13 * Le, 156, -22 * Le],
            [-13 * Le, -3 * Le**2, -22 * Le, 4 * Le**2],
        ],
        dtype=dtype,
        device=device,
    )

    # ------------------------------------------------------------------ #
    # ---------- global matrices initialisation ------------------------ #
    # ------------------------------------------------------------------ #
    K_glb = torch.zeros(samples, dof_total, dof_total, dtype=dtype, device=device)
    M_glb = torch.zeros_like(K_glb)

    # pre-compute DOF indices for each element
    elem_dof = torch.tensor(
        [[2 * i, 2 * i + 1, 2 * (i + 1), 2 * (i + 1) + 1] for i in range(n_elements)],
        dtype=torch.long,
        device=device,
    )

    # ------------------------------------------------------------------ #
    # ---------- assembly loop (vectorised over samples) --------------- #
    # ------------------------------------------------------------------ #
    for e in range(n_elements):
        idx = elem_dof[e]                         # (4,)
        dmg = dmgfields[:, e]                    # (samples,)
        K_e = Ke_base * dmg[:, None, None]        # (samples,4,4)
        mass_coeff = dmg.pow(-mass_dmg_power)
        M_e = Me_base * mass_coeff[:, None, None]

        # add element contributions (broadcasted on batch dim)
        K_glb[:, idx[:, None], idx] += K_e
        M_glb[:, idx[:, None], idx] += M_e

    # enforce symmetry (floating error safeguard)
    K_glb = 0.5 * (K_glb + K_glb.transpose(-1, -2))
    M_glb = 0.5 * (M_glb + M_glb.transpose(-1, -2))

    # ------------------------------------------------------------------ #
    # ---------- static condensation ----------------------------------- #
    # ------------------------------------------------------------------ #
    trans = torch.arange(0, dof_total, 2, device=device)   # w DOFs
    rot   = torch.arange(1, dof_total, 2, device=device)   # θ DOFs

    # block extraction
    Kpp, Kpr, Krp, Krr = (
        K_glb[:, trans][:, :, trans],
        K_glb[:, trans][:, :, rot],
        K_glb[:, rot][:, :, trans],
        K_glb[:, rot][:, :, rot],
    )
    Mpp, Mpr, Mrp, Mrr = (
        M_glb[:, trans][:, :, trans],
        M_glb[:, trans][:, :, rot],
        M_glb[:, rot][:, :, trans],
        M_glb[:, rot][:, :, rot],
    )

    # inverse on rotation sub-blocks
    Krr_inv = torch.linalg.inv(Krr)
    Mrr_inv = torch.linalg.inv(Mrr)

    # condensed matrices
    K_cond = Kpp - Kpr @ Krr_inv @ Krp
    M_cond = Mpp - Mpr @ Mrr_inv @ Mrp

    return M_cond, K_cond

def need_MK_matrices(config):
    """Check if any loss in config needs M/K matrices."""
    # 支持字符串匹配和自定义扩展
    MK_related_loss_keywords = [
        'MO',  # MassOrthogonalityLoss
        'SO',  # StiffnessOrthogonalityLoss
    ]
    for loss in config["loss"]["losses"]:
        if any(key in loss[0] for key in MK_related_loss_keywords):
            return True
    return False

def print_epoch_results(
    ep, time1, time2,
    loss_names, train_losses, valid_losses, loss_weights, weighted_train_loss, weighted_valid_loss,
    evaluations, train_evals, valid_evals
):
    print("-" * 80)
    print(f"Time | Training: {time1:.2f}s | Epoch: {time2:.2f}s\n")

    # Loss表格
    NAME_W = 18
    FIELD_W = 11
    row_fmt = (
        "| {:<16} | {:>9} | {:>9} | {:>9} | {:>9} | {:>9} |"
    )
    sep_line = (
        "+" + "-"*NAME_W +
        "+" + "-"*FIELD_W +
        "+" + "-"*FIELD_W +
        "+" + "-"*FIELD_W +
        "+" + "-"*FIELD_W +
        "+" + "-"*FIELD_W +
        "+"
    )
    print(sep_line)
    print(row_fmt.format("Loss Name", "Train", "Wt", "Wtd Train", "Val", "Wtd Val"))
    print(sep_line)
    for idx, name in enumerate(loss_names):
        print(row_fmt.format(
            name[:NAME_W],
            f"{train_losses[idx]:.4f}",
            f"{loss_weights[idx]:.3f}",
            f"{train_losses[idx]*loss_weights[idx]:.4f}",
            f"{valid_losses[idx]:.4f}",
            f"{valid_losses[idx]*loss_weights[idx]:.4f}",
        ))
    print(sep_line)
    print(row_fmt.format(
        "TOTAL",
        "", "", f"{weighted_train_loss[0]:.4f}", "", f"{weighted_valid_loss[0]:.4f}"
    ))
    print(sep_line + "\n")

    # Eval表格，同样字段设置
    print("Evaluation Metrics:")
    eval_row_fmt = "| {:<16} | {:>9} | {:>9} |"
    eval_sep = "+" + "-"*NAME_W + "+" + "-"*FIELD_W + "+" + "-"*FIELD_W + "+"
    print(eval_sep)
    print(eval_row_fmt.format("Metric", "Train", "Val"))
    print(eval_sep)
    for idx, name in enumerate(evaluations):
        print(eval_row_fmt.format(
            name[:NAME_W],
            f"{train_evals[idx]:.4f}",
            f"{valid_evals[idx]:.4f}"
        ))
    print(eval_sep)
    print("-" * 80)
    print()



def get_system_info(print_info=True):
    """Get system and hardware specifications.

    Args:
        print_info (bool): Whether to print info to console.
    Returns:
        dict: System information dictionary.
    """
    # Collect system information
    system_info = {
        "Device Name": socket.gethostname(),
        "Platform": platform.system(),
        "Release": platform.release(),
        "Version": platform.version(),
        "Architecture": platform.machine(),
        "CPU": platform.processor(),
        "CPU Cores (Logical)": psutil.cpu_count(logical=True),
        "CPU Cores (Physical)": psutil.cpu_count(logical=False),
        "RAM Total": f"{round(psutil.virtual_memory().total / (1024.0 ** 3))} GB",
        "RAM Available": f"{round(psutil.virtual_memory().available / (1024.0 ** 3))} GB",
        "Python Version": sys.version.split()[0]
    }

    # Disk info
    disk_total, _, disk_free = shutil.disk_usage("/")
    system_info["Disk Total"] = f"{disk_total // (1024**3)} GB"
    system_info["Disk Free"] = f"{disk_free // (1024**3)} GB"

    # Torch and CUDA info
    system_info["PyTorch Version"] = torch.__version__
    if torch.cuda.is_available():
        system_info["GPU"] = torch.cuda.get_device_name(0)
        system_info["GPU Count"] = torch.cuda.device_count()
        system_info["CUDA Version"] = torch.version.cuda
    else:
        system_info["GPU"] = "None"
        system_info["GPU Count"] = 0
        system_info["CUDA Version"] = "N/A"

    # Print if required
    if print_info:
        line_width = 80
        print("=" * line_width)
        title = " System and Hardware Specifications "
        print(title.center(line_width, "="))
        print("-" * line_width)
        for k, v in system_info.items():
            print(f"{k:<22}: {v}")
        print("=" * line_width)

    return system_info