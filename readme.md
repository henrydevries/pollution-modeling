# Source code for "Modeling Spatial Pollution Dynamics under Irregular Sampling, Missing Values, and Outliers"


Authors: Henry de Vries, Yicong Lin, and Andre Lucas

This repository contains the code used to produce the simulation results and empirical findings presented in the paper Modeling Spatial Pollution Dynamics under Irregular Sampling, Missing Values, and Outliers. For any questions feel free to contact the authors at [henryjcdevries@outlook.com](mailto:henryjcdevries@outlook.com).



## Repository Structure
### Section 3: Simulations
The code for reproducing the simulation study in Section 3 is located in the `simulations/` directory.

The main components are:

- `simulationMain.py` — Main orchestration script controlling all simulation settings and model specifications. This is the primary entry point for running simulations.
- `surfaceFactorDGP.py` - Generation of data.
- Model specifications are implemented in separate modules:
   `fGAS.py`, `pooling.py`, `rollingaverage.py`, `statespace.py`, and `VARL1.py`.
- The FPCA basis is constructed by `FPCA.py`.
- The eigenfunctions of the empirical section are stored in `Q_logdata.npy`, which are used for the DGP of the simulation.

All simulations were executed on the Snellius high-performance computing system ([SURF Snellius](https://www.surf.nl/diensten/rekenen/snellius-de-nationale-supercomputer)). The simulations were run using Python 3.9.21 with SciPy 1.13.1. As a result, some legacy numerical functions are used (e.g., `np.trapz`).

### Section 4: Empirical application to PM2.5 forecasting
All code for the empirical application is located in the `empirical/` directory. This code is run **locally**.

The empirical analysis uses PM2.5 data obtained from [PurpleAir](https://www.purpleair.com/). Data loading is handled by `dataloader` function defined in the respective notebooks.

- Data exploration (Section 4.1) is performed in `empirical/data interpretation and exploration/dataset_exploration.ipynb`.
- PC interpretation (Section 4.2) is performed in `empirical/data interpretation and exploration/PC-interpretation.ipynb`.
- All forecasting plots in Section 4.3 and Appendix B.2 are produced in `empirical/local`.
- The FPCA basis is again constructed using `FPCA.py`.

#### Monte Carlo Assessment
The Monte Carlo assessment based on the empirical PM2.5 data follows a similar structure to the simulation study in Section 3. The corresponding code is located in `empirical/sensorsamplings/`. This code is again run on the Snellius high-performance computer.

- `orchestrator.py` — Main orchestration script controlling all simulation settings and model specifications. This is the primary entry point for running simulations.
- `dataload.py`  — Loading of the dataset
- Model specifications are implemented in separate modules:
   `fGAS.py`, `pooling.py`, `rollingaverage.py`, `statespace.py`, and `VARL1.py`.
- The FPCA basis is constructed by `FPCA.py`.