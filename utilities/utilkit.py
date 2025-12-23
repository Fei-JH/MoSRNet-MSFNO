# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:42:26 2024

@author: FEI JINGHAO
"""
import importlib
import os
import random
import shutil
import socket
import sys
import platform

import numpy as np
import psutil
import torch


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_loss_function(module_path: str):
    """
    Dynamically import a loss function or class from a module path.

    Example:
        'losses.MAE.MAE' -> losses/mae.py -> MAE
    """
    try:
        *module_parts, function_name = module_path.split(".")
        module_parts[1] = module_parts[1].lower()
        module_name = ".".join(module_parts)
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ImportError(f"Module '{module_name}' not found.") from exc

    try:
        func = getattr(module, function_name)
    except AttributeError as exc:
        raise AttributeError(
            f"Module '{module_name}' does not contain '{function_name}'."
        ) from exc

    if not callable(func):
        raise TypeError(
            f"The attribute '{function_name}' in module '{module_name}' is not callable."
        )

    return func


def safe_path(path: str) -> str:
    """Convert a path to an absolute Windows-safe path with a long-path prefix."""
    abs_path = os.path.abspath(path)
    return r"\\?\{}".format(abs_path)


def print_epoch_results(
    ep,
    time1,
    time2,
    loss_names,
    train_losses,
    valid_losses,
    loss_weights,
    weighted_train_loss,
    weighted_valid_loss,
    evaluations,
    train_evals,
    valid_evals,
):
    """Print a formatted summary for the current epoch."""
    print("-" * 80)
    print(f"Time | Training: {time1:.2f}s | Epoch: {time2:.2f}s\n")

    name_width = 18
    field_width = 11
    row_fmt = "| {:<16} | {:>9} | {:>9} | {:>9} | {:>9} | {:>9} |"
    sep_line = (
        "+" + "-" * name_width
        + "+" + "-" * field_width
        + "+" + "-" * field_width
        + "+" + "-" * field_width
        + "+" + "-" * field_width
        + "+" + "-" * field_width
        + "+"
    )
    print(sep_line)
    print(row_fmt.format("Loss Name", "Train", "Wt", "Wtd Train", "Val", "Wtd Val"))
    print(sep_line)
    for idx, name in enumerate(loss_names):
        print(
            row_fmt.format(
                name[:name_width],
                f"{train_losses[idx]:.4f}",
                f"{loss_weights[idx]:.3f}",
                f"{train_losses[idx] * loss_weights[idx]:.4f}",
                f"{valid_losses[idx]:.4f}",
                f"{valid_losses[idx] * loss_weights[idx]:.4f}",
            )
        )
    print(sep_line)
    print(
        row_fmt.format(
            "TOTAL",
            "",
            "",
            f"{weighted_train_loss[0]:.4f}",
            "",
            f"{weighted_valid_loss[0]:.4f}",
        )
    )
    print(sep_line + "\n")

    print("Evaluation Metrics:")
    eval_row_fmt = "| {:<16} | {:>9} | {:>9} |"
    eval_sep = "+" + "-" * name_width + "+" + "-" * field_width + "+" + "-" * field_width + "+"
    print(eval_sep)
    print(eval_row_fmt.format("Metric", "Train", "Val"))
    print(eval_sep)
    for idx, name in enumerate(evaluations):
        print(
            eval_row_fmt.format(
                name[:name_width],
                f"{train_evals[idx]:.4f}",
                f"{valid_evals[idx]:.4f}",
            )
        )
    print(eval_sep)
    print("-" * 80)
    print()


def get_system_info(print_info=True):
    """Get system and hardware specifications."""
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
        "Python Version": sys.version.split()[0],
    }

    disk_total, _, disk_free = shutil.disk_usage("/")
    system_info["Disk Total"] = f"{disk_total // (1024 ** 3)} GB"
    system_info["Disk Free"] = f"{disk_free // (1024 ** 3)} GB"

    system_info["PyTorch Version"] = torch.__version__
    if torch.cuda.is_available():
        system_info["GPU"] = torch.cuda.get_device_name(0)
        system_info["GPU Count"] = torch.cuda.device_count()
        system_info["CUDA Version"] = torch.version.cuda
    else:
        system_info["GPU"] = "None"
        system_info["GPU Count"] = 0
        system_info["CUDA Version"] = "N/A"

    if print_info:
        line_width = 80
        print("=" * line_width)
        title = " System and Hardware Specifications "
        print(title.center(line_width, "="))
        print("-" * line_width)
        for key, value in system_info.items():
            print(f"{key:<22}: {value}")
        print("=" * line_width)

    return system_info
