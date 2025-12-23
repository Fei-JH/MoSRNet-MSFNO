<!--
 * @Author: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
 * @Date: 2025-12-23 18:04:08
 * @LastEditors: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
 * @LastEditTime: 2025-12-23 19:00:45
-->
# Lightweight Modal Super-Resolution CNN and Fourier Neural Operator for High-Resolution Damage Identification

## Overview
This repository implements a synergistic framework for high-resolution damage identification (stiffness-loss field inference) for beam-/bridge-type structures:

1) **MoSRNet**: super-resolves **coarsely sampled** mode shapes to a fine mesh.  
2) **MS-FNO**: learns an **operator mapping** from modal fields to the **stiffness field** (damage field).  
3) **ResNet (1D)**: baseline model for comparison.

**Highlights:**
1) The MoSRNet–MSFNO framework enables **high-resolution structural damage identification**.
2) Fourier Neural Operator **learns the inverse mapping** and exceeds the ResNet baseline.
3) Gaussian Random Fields **generates continuous damage fields** for better generalization. 
4) Reliable damage identification achieved **using only three modes and seven sensors**.
5) Experiments confirm **accurate and stable identification** across multiple damage cases.

## Network Architecture
![Network architecture](docs/figs/fig_framework.png)

## Technical Pipeline
![Technical route](docs/figs/fig_pipeline.png)


## Data and setup
### FE model 
An Euler–Bernoulli beam FE model is used (540 elements / 541 nodes). Key parameters:
- Length = 5.4 m  
- E = 2.1e11 N/m²  
- I = 5.709e-7 m⁴  
- ρ = 7850 kg/m³  
- A = 6.542e-3 m²  

### Datasets
Datasets are organized under `datasets/` (naming may follow the manuscript):

- **BeamDI-Num**: 9000 numerical samples (8000 train / 1000 val) for **MS-FNO** and **ResNet**.
- **BeamDI-Num(DS)**: paired **coarse–fine** mode shapes for training **MoSRNet**.  
  - Coarse sampling corresponds to the experimental sensing layout:
    - **Location of 7 accelerometers** along the span, and (when forming a fixed-length coarse vector for learning) **two boundary points** can be included as endpoints, yielding **9 spatial points** (consistent with a `(3, 9)` MoSRNet input).
- **BeamDI-Phy**: experimental mode shapes extracted by **BAYOMA**. 4 scenarios included.

## Repository structure
- `models/`: MoSRNet, MS-FNO, ResNet definitions  
- `experiments/`: training loops and metrics  
- `configs/`: training configs (paper hyperparameters)  
- `datasets/`: numerical and experimental `.pt` data  
- `results/`: checkpoints and postprocessed outputs  
- `postprocess_*.py`: figures and metrics for the manuscript  
- `run_train_*.py`, `unified_training_manager.py`: training entry points  

## Quick start
Install dependencies:
```bash
pip install -r requirements.txt
```

Train models (reads configs under `configs/`):
```bash
python run_train_mosrnet.py
python run_train_msfno.py
python run_train_resnet.py
```

Batch training (interactive or CLI):
```bash
python unified_training_manager.py
# or: python unified_training_manager.py --scripts all --configs-all all --wandb 0
```

OOD / Monte Carlo Simulation (damage localization):
```bash
python run_mcs.py
```
Typical output (CSV): `results/mcs_test/mcs_maxpos_rawidx.csv`

Postprocess figures and metrics:
```bash
python postprocess_c3s4_fig.py
python postprocess_c4s1_fig.py
python postprocess_c4s2_csv.py
python postprocess_c4s2_fig.py
python postprocess_c4s3_fig.py
python postprocess_c5s2_fig.py
python postprocess_c5s3_fig.py
```

## Experimental setup
A simply supported steel beam (span 5.4 m) is tested with two damage regions:

- **DMG1**: multiple vertical cuts at the left side bottom flange  
- **DMG2**: wedge-shaped cut at the right side bottom flange  

By adding/removing reinforcement plates, four scenarios are created: **MCUT**, **WEDG**, **WCUT**, and **REIN**.  
**REIN** is used as the **base state** for bias calibration. Mode shapes are identified from acceleration data using **BAYOMA** (with repeated measurements per scenario, then averaged as the representative mode shape).

![Experimental setup](docs/figs/fig_experiment_setup.png)
