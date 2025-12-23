"""
Author: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
Date: 2025-08-12 18:06:32
LastEditors: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
LastEditTime: 2025-08-22 16:36:26
"""

import numpy as np
from scipy.interpolate import interp1d


def generate_sequence(length=540, y=10, noise_range=0.05, dip_range=(0.15, 0.6), seed=None):
    """
    Generate a 1D sequence with a random dip and linear transitions.

    Returns:
        seq (np.ndarray): Generated sequence.
        x (int): Dip position.
        m (float): Dip magnitude.
    """
    if seed is not None:
        np.random.seed(seed)

    seq = np.ones(length)
    x = np.random.randint(0, length)
    m = np.random.uniform(*dip_range)
    seq[x] = 1 - m

    left_start = max(0, x - y)
    left_end = x
    if left_end > left_start:
        seq[left_start:left_end] = np.linspace(1, 1 - m, left_end - left_start, endpoint=False)

    right_start = x + 1
    right_end = min(length, x + y + 1)
    if right_end > right_start:
        seq[right_start:right_end] = np.linspace(1 - m, 1, right_end - right_start, endpoint=False)

    mask = np.ones(length, dtype=bool)
    mask[left_start:right_end] = False
    noise = np.random.uniform(-noise_range, noise_range, length)
    seq[mask] += noise[mask]

    return seq, x, m


def interpolate_1d(arr, tgt):
    """Interpolate a 1D array to a target length using linear interpolation."""
    ori_len = len(arr)
    x_old = np.linspace(0, 1, ori_len)
    f = interp1d(x_old, arr, kind="linear")
    x_new = np.linspace(0, 1, tgt)
    return f(x_new)
