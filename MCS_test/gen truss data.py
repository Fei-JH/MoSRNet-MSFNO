# -*- coding: utf-8 -*-
"""
Created on Tue May 14 14:43:40 2024

@author: Owner
"""
import numpy as np
import os
import time
import pandas as pd 
import FE
from tqdm import tqdm
import matplotlib as plt
#%%
# Setting up the directory
scenario = "BEAM01_VaryingLn0"
save_dir =fr"..\Dataset\{scenario}"
os.makedirs(save_dir, exist_ok=True)

# Loading damage data
damage_file_path = r"D:\FEI JINGHAO\OneDrive\OneDrive - Kyoto University\Research\ModalStiff FNO\GEN GRF\DMG_random_l_20250619_161849.csv"
dmg = np.array(pd.read_csv(damage_file_path, header=None, skiprows=0))
# dmg = np.zeros((5500, 540))
# dmg=dmg_temp


# Beam properties
L = 5.4
E = 210e9
I = 57.48e-8
rho = 7850
A = 65.423 * 0.0001
n_elements = 540

# Preallocate arrays for results
modes = np.empty((len(dmg), 4, n_elements + 1))
freq = np.empty((len(dmg), 4))

# Start timing
start_time = time.time()

for i in tqdm(range(len(dmg))):
    beam = FE.BeamAnalysis(L, E, I, rho, A, n_elements)
    beam.assemble_matrices(dmgfield=dmg[i, :], mass_dmg_power=0)
    beam.apply_BC()
    frequencies, eigenvectors = beam.solve_eigenproblem()
    
    # Store frequencies
    freq[i, :] = frequencies[2:6]
    
    # Store mode shapes (only interested in modes 2-6)
    u_vectors, _ = beam.split_eigenvectors(eigenvectors)
    modes[i, :, :] = u_vectors[:, 2:6].T

# End timing
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)

print(f"Time taken for loop execution: {minutes} min {seconds} s")

# Save results
np.savetxt(os.path.join(save_dir, "DMG.csv"), dmg, delimiter=",")
for mode_idx in range(4):
    np.savetxt(os.path.join(save_dir, f"MODE{mode_idx + 1}.csv"), modes[:, mode_idx, :], delimiter=",")
np.savetxt(os.path.join(save_dir, "FREQ.csv"), freq, delimiter=",")