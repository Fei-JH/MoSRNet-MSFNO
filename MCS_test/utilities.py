import numpy as np

import numpy as np

def generate_sequence(length=540, y=10, noise_range=0.05, dip_range=(0.15, 0.6), seed=None):
    """
    Generate a 1D sequence with a random dip and linear transitions, plus noise elsewhere.
    
    Parameters
    ----------
    length : int
        Length of the sequence.
    y : int
        Number of points before and after the dip for linear transition.
    noise_range : float
        The absolute range of random noise added to non-transition regions (default [-0.05, 0.05]).
    dip_range : tuple(float, float)
        Range of dip magnitude m, where the dip value is 1-m at the chosen point.
    seed : int or None
        Random seed for reproducibility.
    
    Returns
    -------
    seq : np.ndarray
        The generated 1D sequence.
    x : int
        The position of the dip.
    m : float
        The dip magnitude (actual value subtracted from 1 at the dip).
    """
    if seed is not None:
        np.random.seed(seed)
    
    seq = np.ones(length)
    x = np.random.randint(0, length)  # Allow dip anywhere
    m = np.random.uniform(*dip_range)
    seq[x] = 1 - m

    # Left transition: from 1 to 1-m
    left_start = max(0, x - y)
    left_end = x  # not inclusive
    if left_end > left_start:
        seq[left_start:left_end] = np.linspace(1, 1 - m, left_end - left_start, endpoint=False)

    # Right transition: from 1-m to 1
    right_start = x + 1
    right_end = min(length, x + y + 1)
    if right_end > right_start:
        seq[right_start:right_end] = np.linspace(1 - m, 1, right_end - right_start, endpoint=False)
    
    # Add noise to non-transition regions
    mask = np.ones(length, dtype=bool)
    mask[left_start:right_end] = False
    noise = np.random.uniform(-noise_range, noise_range, length)
    seq[mask] += noise[mask]
    
    return seq, x, m


# Example usage:
# seq, x, m = generate_sequence()
