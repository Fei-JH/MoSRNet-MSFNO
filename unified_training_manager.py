'''
Author: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
Date: 2025-08-14 12:00:18
LastEditors: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
LastEditTime: 2025-08-14 14:36:14
'''


# -*- coding: utf-8 -*-
"""
Controller (import-based) with CLI args:
- One-line run with fixed choices via argparse
- Or interactive mode when args missing
- Pre-select all configs, then execute tasks sequentially
- Four-line header per task; total elapsed printed at the end
- Fixed print width: 80 columns
- Runs inside CURRENT interpreter (no interpreter switching)
"""

import os
import sys
import yaml
import time
import textwrap
import importlib
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

WIDTH = 80
CONFIG_DIR = Path("./configs")

# Registry: (display_name, module_name (without .py), config_prefix)
SCRIPTS: List[Tuple[str, str, str]] = [
    ("MoSRNet", "run_train_mosrnet", "mosrnet"),
    ("MSFNO",   "run_train_msfno",   "msfno"),
    ("ResNet",  "run_train_resnet",  "resnet"),
]

# ------------------------ UI helpers (80-col printing) ------------------------ #
def print_divider(char: str = "=") -> None:
    """Print a full-width divider line."""
    print(char * WIDTH)

def print_wrapped(msg: str) -> None:
    """Print a message wrapped to WIDTH columns."""
    if not msg:
        print("")
        return
    for line in textwrap.fill(msg, width=WIDTH).splitlines():
        print(line)

def input_line(prompt: str) -> str:
    """Prompt for input (single line)."""
    print_wrapped(prompt)
    return input("> ").strip()

# ----------------------------- selection parsers ----------------------------- #
def parse_indices_string(raw: str, n: int) -> List[int]:
    """
    Parse a selection string like:
      - 'all' -> [1..n]
      - '1,3,5'
      - '2-6'
      - '1,3-5,8'
    Return 1-based sorted unique indices.
    """
    s = raw.strip().lower()
    if s == "all":
        return list(range(1, n + 1))
    parts = [p.strip() for p in s.split(",") if p.strip()]
    result = set()
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            if not (a.isdigit() and b.isdigit()):
                raise ValueError(f"Invalid range: {p}")
            ai, bi = int(a), int(b)
            if ai < 1 or bi < 1 or ai > n or bi > n or ai > bi:
                raise ValueError(f"Range out of bounds: {p} (1..{n})")
            for k in range(ai, bi + 1):
                result.add(k)
        else:
            if not p.isdigit():
                raise ValueError(f"Non-numeric index: {p}")
            v = int(p)
            if v < 1 or v > n:
                raise ValueError(f"Index out of range: {v} (1..{n})")
            result.add(v)
    if not result:
        raise ValueError("Empty selection.")
    return sorted(result)

def parse_scripts_arg(raw: str) -> List[int]:
    """Parse --scripts against the SCRIPTS registry."""
    if raw.strip().lower() == "all":
        return [1, 2, 3]
    # Validate indices 1..len(SCRIPTS)
    return parse_indices_string(raw, len(SCRIPTS))

# --------------------------- config scanning helpers -------------------------- #
def scan_configs(prefix: str) -> List[str]:
    """Scan CONFIG_DIR for YAML/YML files that start with the given prefix."""
    files: List[str] = []
    if CONFIG_DIR.is_dir():
        for f in CONFIG_DIR.iterdir():
            if f.is_file() and f.suffix.lower() in (".yaml", ".yml") and f.name.startswith(prefix):
                files.append(f.name)
    return sorted(files)

# --------------------------- interactive selection --------------------------- #
def select_scripts_interactive() -> List[int]:
    print_divider("=")
    print_wrapped("Available training scripts:")
    for i, (name, module_name, prefix) in enumerate(SCRIPTS, start=1):
        print_wrapped(f"{i}. {name} [{module_name}.py] (config prefix: {prefix})")
    print_wrapped("Enter indices separated by comma, ranges (e.g., 1,3-5), or 'all'.")
    while True:
        raw = input_line("Choose scripts:")
        try:
            return parse_scripts_arg(raw)
        except Exception as e:
            print_wrapped(f"Error: {e}")

def select_configs_for_prefix_interactive(prefix: str, title: str) -> List[str]:
    print_divider("=")
    print_wrapped(f"Scanning configs for {title} (prefix: '{prefix}')")
    files = scan_configs(prefix)
    if not files:
        print_wrapped("No matching config files found.")
        return []

    for i, fn in enumerate(files, start=1):
        print_wrapped(f"{i}. {fn}")
    print_wrapped("Enter indices/ranges (e.g., 1,2,5 or 2-6) or 'all'.")

    while True:
        raw = input_line(f"Choose configs for {title}:")
        try:
            idxs = parse_indices_string(raw, len(files))
            return [files[i - 1] for i in idxs]
        except Exception as e:
            print_wrapped(f"Error: {e}")

def detect_and_pick_wandb_interactive() -> bool:
    print_divider("=")
    # Try import
    has_wandb = False
    try:
        import wandb  # noqa: F401
        has_wandb = True
    except Exception:
        has_wandb = False

    if not has_wandb:
        print_wrapped("wandb is NOT installed in the current interpreter. It will be disabled.")
        return False

    print_wrapped("wandb is installed. Choose usage:")
    print_wrapped("1. Enable")
    print_wrapped("2. Disable")
    while True:
        choice = input_line("Enter 1 / 2:").lower()
        if choice in {"1", "2"}:
            return choice == "1"
        print_wrapped("Invalid choice. Please enter 1 or 2.")

# ----------------------- task building & execution ----------------------- #
def build_tasks(selected_scripts: List[int],
                configs_map: Dict[str, str] = None,
                configs_all: str = None) -> List[Tuple[str, str, str]]:
    """
    Build the task list:
      - selected_scripts: list of script indices (1-based)
      - configs_map: optional mapping from prefix -> selection string ('all', '1,3-5', etc.)
      - configs_all: optional default selection string applied to all selected scripts
    Returns: list of (module_name, script_display_name, config_filename)
    """
    tasks: List[Tuple[str, str, str]] = []
    for idx in selected_scripts:
        name, module_name, prefix = SCRIPTS[idx - 1]
        files = scan_configs(prefix)
        if not files:
            continue

        # Determine selection string for this prefix
        sel_raw = None
        if configs_map and prefix in configs_map and configs_map[prefix]:
            sel_raw = configs_map[prefix]
        elif configs_all:
            sel_raw = configs_all

        if sel_raw is None:
            # No CLI selection provided -> interactive selection for this script
            chosen_files = select_configs_for_prefix_interactive(prefix, name)
        else:
            # Parse indices against scanned files
            idxs = parse_indices_string(sel_raw, len(files))
            chosen_files = [files[i - 1] for i in idxs]

        for cfg in chosen_files:
            tasks.append((module_name, name, cfg))
    return tasks

def run_tasks(tasks: List[Tuple[str, str, str]], use_wandb: bool) -> None:
    """
    Run all tasks sequentially. Before each task, print four lines:
      1) total tasks and current task index
      2) elapsed time since controller start
      3) current script name
      4) current config file name
    """
    start_time = time.time()

    # Pre-import torch once to avoid repeated overhead; still safe to import per-task too.
    try:
        import torch
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    except Exception as e:
        device = None
        print_wrapped(f"Warning: torch init failed; device unknown: {e}")

    total = len(tasks)
    for i, (module_name, script_name, cfg_name) in enumerate(tasks, start=1):
        print_divider("=")
        print_wrapped(f"Tasks: {total} | Current: {i}")
        elapsed = time.time() - start_time
        print_wrapped(f"Elapsed: {elapsed:.1f} s")
        print_wrapped(f"Script: {script_name} [{module_name}.py]")
        print_wrapped(f"Config: {cfg_name}")
        print_divider("-")

        # Import module and resolve required symbols
        try:
            mod = importlib.import_module(module_name)
        except Exception as e:
            print_wrapped(f"Failed to import {module_name}: {e}")
            continue

        try:
            run_train_1d = getattr(mod, "run_train_1d")
            model_classes = getattr(mod, "model_classes")
        except Exception as e:
            print_wrapped(f"Module '{module_name}' lacks required symbols: {e}")
            continue

        # Load YAML
        cfg_path = CONFIG_DIR / cfg_name
        if not cfg_path.is_file():
            print_wrapped(f"Config file not found: {cfg_path}")
            continue

        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                config = yaml.load(f, Loader=yaml.SafeLoader)
        except Exception as e:
            print_wrapped(f"Failed to load YAML '{cfg_name}': {e}")
            continue

        # Resolve model class
        try:
            model_key = config["model"]["model"]
            model_class = model_classes[model_key]
        except Exception as e:
            print_wrapped(f"Failed to get model class from config['model']['model']: {e}")
            continue

        # Prepare device if not already
        if device is None:
            try:
                import torch  # lazy import fallback
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            except Exception as e:
                print_wrapped(f"Failed to initialize torch device: {e}")
                continue

        # Run training
        try:
            _ = run_train_1d(
                config=config,
                config_name=cfg_name,
                device=device,
                model_class=model_class,
                use_wandb=use_wandb,
                sweep=False
            )
        except Exception as e:
            print_wrapped(f"Training failed for '{cfg_name}' in module '{module_name}': {e}")

    total_elapsed = time.time() - start_time
    print_divider("=")
    print_wrapped(f"All tasks finished. Total elapsed: {total_elapsed:.1f} s")

# --------------------------------- argparse --------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unified Training Manager",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # scripts: 'all' or indices/ranges
    p.add_argument("--scripts", type=str, default=None,
                   help="Script selection: 'all' or indices/ranges, e.g. '1,3' or '1-3'")

    # wandb: 1/0
    p.add_argument("--wandb", type=int, choices=[0, 1], default=None,
                   help="Enable wandb (1) or disable (0)")

    # configs-all: default selection for ALL selected scripts (if specific not provided)
    p.add_argument("--configs-all", type=str, default=None,
                   help="Default config selection for all selected scripts: 'all' or indices/ranges")

    # per-script overrides (by prefix)
    p.add_argument("--mosrnet-configs", type=str, default=None,
                   help="MoSRNet configs selection: 'all' or indices/ranges, e.g. '1,3-5'")
    p.add_argument("--msfno-configs", type=str, default=None,
                   help="MSFNO configs selection: 'all' or indices/ranges")
    p.add_argument("--resnet-configs", type=str, default=None,
                   help="ResNet configs selection: 'all' or indices/ranges")
    return p

# ----------------------------------- main ----------------------------------- #
def main() -> None:
    print_divider("=")
    print_wrapped("Unified Training Manager")
    print_divider("=")

    # Parse CLI args
    parser = build_arg_parser()
    args = parser.parse_args()

    # Select scripts (CLI or interactive)
    if args.scripts is not None:
        try:
            selected_scripts = parse_scripts_arg(args.scripts)
        except Exception as e:
            print_wrapped(f"Invalid --scripts: {e}")
            sys.exit(1)
    else:
        selected_scripts = select_scripts_interactive()

    # wandb selection (CLI or interactive)
    if args.wandb is not None:
        use_wandb_flag = bool(args.wandb)
    else:
        use_wandb_flag = detect_and_pick_wandb_interactive()

    # Build config selection map
    configs_map: Dict[str, str] = {}
    if args.mosrnet_configs: configs_map["mosrnet"] = args.mosrnet_configs
    if args.msfno_configs:   configs_map["msfno"]   = args.msfno_configs
    if args.resnet_configs:  configs_map["resnet"]  = args.resnet_configs

    # Build all tasks upfront
    tasks = build_tasks(
        selected_scripts=selected_scripts,
        configs_map=configs_map if configs_map else None,
        configs_all=args.configs_all,
    )
    if not tasks:
        print_wrapped("No tasks to run. Exiting.")
        return

    # Execute
    run_tasks(tasks, use_wandb_flag)

if __name__ == "__main__":
    main()
