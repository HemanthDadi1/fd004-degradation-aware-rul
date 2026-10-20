# Data

The NASA C-MAPSS turbofan degradation data are not redistributed here. Download them from the
NASA Prognostics Center of Excellence data repository ("Turbofan Engine Degradation Simulation",
Saxena et al., 2008) and place the files so that this path exists:

    CMAPSSData/train_FD004.txt
    CMAPSSData/train_FD002.txt   (FD002, notebooks 16–18)

Only these two training files are used. Set the environment variable `FD004_ROOT` to the repository root.

Included inputs:

- `artifacts/FD004_baseline_tensors.npz` (3.8 MB): 30-cycle IID windows from Part I notebook 01.
- `artifacts/P2_E8_60cycle_tensors.npz` (4.8 MB): 60-cycle IID windows from the original Paper 2
  notebooks, checked by notebooks 09 and 12 to align exactly with the Part I IID split.
- `FD004_DS_B_Results/`, `FD004_DS_A_Rerun_Results/engine_split.csv`, `models/b6_variants/`:
  split and threshold files the notebooks verify against.
