import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["MKL_DYNAMIC"] = "FALSE"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"
os.environ["GOTO_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"

import numpy as np
from scipy import stats
from surfaceFactorDGP import surfacepicker
from FPCA import FPCA
import rollingaverage
import statespace
import basisfunctions
import fGAS
import pooling
import VARL1
from joblib import Parallel, delayed
import pandas as pd


def MCrun(p, q, T, N, sampling, r, r0, nu, Omega0, R):
    if sampling == "uniform":
        s1 = np.random.rand(N)
        s2 = np.random.rand(N)
    elif sampling == "beta":
        s1 = stats.beta.rvs(a=5, b=2, size=N)
        s2 = stats.beta.rvs(a=5, b=2, size=N)
    else:
        raise ValueError("Error in Sampling Distribution")

    Phi, Phi_surface, Y_list, Mu_list, Mu_list_surface = surfacepicker(s1, s2, T, r0, p, q, nu)

    train_size = T - 1000
    Y_list_train = Y_list[:train_size]

    _, Phi_eig, Phi_surface_eig, Muhat, Muhat_surface = FPCA(Y_list_train, s1, s2, r, Phi_surface, Omega0, R)
    Muhat_list = np.tile(Muhat, (T, 1))
    Muhat_list_surface = np.tile(Muhat_surface, (T, 1))

    Y_list = Y_list - Muhat_list
    Mu_list = Mu_list - Muhat_list
    Mu_list_surface = Mu_list_surface - Muhat_list_surface
    Y_list_train = Y_list[:train_size]
    Y_list_test = Y_list[train_size:]
    Mu_list_test = Mu_list[train_size:]

    model_functions = {
        "fGAS": fGAS.fGAS,
        "rollingaverage": rollingaverage.rollingaverage,
        "statespace": statespace.statespace,
        "pooledgammahat (lags=20)": pooling.pooling,
        "VAR(1)" : VARL1.VARL1
    }

    results = {}

    for name, func in model_functions.items():
        MAE, MAE_surface, MedAE, MedAE_surface = func(
            Phi, Phi_surface,
            Y_list_train, Y_list_test,
            Mu_list_test, Mu_list_surface,
            R
        )

        MAE_dimred, MAE_surface_dimred, MedAE_dimred, MedAE_surface_dimred = func(
            Phi_eig, Phi_surface_eig,
            Y_list_train, Y_list_test,
            Mu_list_test, Mu_list_surface,
            R
        )

        results[name] = {
            "MAE": MAE,
            "MAE_surface": MAE_surface,
            "MAE_dimred": MAE_dimred,
            "MAE_surface_dimred": MAE_surface_dimred,
            "MedAE" : MedAE,
            "MedAE_surface" : MedAE_surface,
            "MedAE_dimred": MedAE_dimred,
            "MedAE_surface_dimred" : MedAE_surface_dimred 
        }

    return results

if __name__ == "__main__":
   M = 500

   param_sets = [
        {
            "scenario": "S1_clean_gaussian",
            "T": 1500,
            "N": 100,
            "p": 1.0,
            "q": 0.0,
            "sampling": "uniform",
            "nu": 500,
            "description": "Clean Gaussian benchmark",
        },
        {
            "scenario": "S2_canonical_contaminated",
            "T": 1500,
            "N": 100,
            "p": 1.0,
            "q": 0.0,
            "sampling": "uniform",
            "nu": 5,
            "description": "Canonical contaminated baseline",
        },
        {
            "scenario": "S3_irregular_sampling",
            "T": 1500,
            "N": 100,
            "p": 1.0,
            "q": 0.0,
            "sampling": "beta",
            "nu": 5,
            "description": "Irregular sampling",
        },
        {
            "scenario": "S4_more_sensors",
            "T": 1500,
            "N": 200,
            "p": 1.0,
            "q": 0.0,
            "sampling": "uniform",
            "nu": 5,
            "description": "Larger spatial sample",
        },
        {
            "scenario": "S5_longer_series",
            "T": 2500,
            "N": 100,
            "p": 1.0,
            "q": 0.0,
            "sampling": "uniform",
            "nu": 5,
            "description": "Longer estimation sample",
        },
        {
            "scenario": "S6_mild_missingness",
            "T": 1500,
            "N": 100,
            "p": 0.9,
            "q": 0.3,
            "sampling": "uniform",
            "nu": 5,
            "description": "Mild missingness: 12.5 percent",
        },
        {
            "scenario": "S7_moderate_missingness",
            "T": 1500,
            "N": 100,
            "p": 0.9,
            "q": 0.6,
            "sampling": "uniform",
            "nu": 5,
            "description": "Moderate missingness: 20 percent",
        },
        {
            "scenario": "S8_severe_missingness",
            "T": 1500,
            "N": 100,
            "p": 0.9,
            "q": 0.9,
            "sampling": "uniform",
            "nu": 5,
            "description": "Severe missingness: 50 percent",
        },
        {
            "scenario": "S9_severe_missing_gaussian",
            "T": 1500,
            "N": 100,
            "p": 0.9,
            "q": 0.9,
            "sampling": "uniform",
            "nu": 500,
            "description": "Severe missingness without contamination",
        },
        {
            "scenario": "S10_combined_stress",
            "T": 1500,
            "N": 100,
            "p": 0.9,
            "q": 0.9,
            "sampling": "beta",
            "nu": 5,
            "description": "Irregular sampling and severe missingness",
        },
        {
            "scenario": "S11_combined_stress_more_sensors",
            "T": 1500,
            "N": 200,
            "p": 0.9,
            "q": 0.9,
            "sampling": "beta",
            "nu": 5,
            "description": "Combined stress with more sensors",
        },
        {
            "scenario": "S12_combined_stress_long_series",
            "T": 2500,
            "N": 100,
            "p": 0.9,
            "q": 0.9,
            "sampling": "beta",
            "nu": 5,
            "description": "Combined stress with longer history",
        }
    ]

   Omega0, R = basisfunctions.basismatrixcomputer()

   rows = []
   for params in param_sets:
       scenario = params["scenario"]
       description = params["description"]
       T = params["T"]
       N = params["N"]
       p = params["p"]
       q = params["q"]
       sampling = params["sampling"] 
       nu = params["nu"]


       mc_results = Parallel(n_jobs=192, verbose=10, backend='multiprocessing')(
           delayed(MCrun)(p, q, T, N, sampling=sampling, r=5, r0=5, nu=nu, Omega0=Omega0, R=R) for _ in range(M)
       )

       for mc_id, res in enumerate(mc_results):
           for model_name, metrics in res.items():
               rows.append({
                   "scenario": scenario,
                   "description": description,
                   "T": T,
                   "N": N,
                   "p": p,
                   "q": q,
                   "missing_rate": (
                    0.0 if (p, q) == (1.0, 0.0)
                    else (1.0 - p) / (2.0 - p - q)),
                   "sampling": sampling,
                   "nu" : nu,
                   "model": model_name,
                   "MAE": metrics["MAE"],
                   "MAE_surface": metrics["MAE_surface"],
                   "MAE_dimred": metrics["MAE_dimred"],
                   "MAE_surface_dimred": metrics["MAE_surface_dimred"],
                   "MedAE": metrics["MedAE"],
                   "MedAE_surface": metrics["MedAE_surface"],
                   "MedAE_dimred": metrics["MedAE_dimred"],
                   "MedAE_surface_dimred": metrics["MedAE_surface_dimred"]
               })

   df_results = pd.DataFrame(rows)

   summary = (
       df_results.groupby(["T", "N", "p", "q", "missing_rate", "sampling", "nu", "model"]).agg(
           MAE_mean=("MAE", "mean"),
           MAE_median=("MAE", "median"),
           MAE_surface_mean=("MAE_surface", "mean"),
           MAE_surface_median=("MAE_surface", "median"),
           MAE_dimred_mean=("MAE_dimred", "mean"),
           MAE_dimred_median=("MAE_dimred", "median"),
           MAE_surface_dimred_mean=("MAE_surface_dimred", "mean"),
           MAE_surface_dimred_median=("MAE_surface_dimred", "median"),
           MedAE_mean=("MedAE", "mean"),
           MedAE_median=("MedAE", "median"),
           MedAE_surface_mean=("MedAE_surface", "mean"),
           MedAE_surface_median=("MedAE_surface", "median"),
           MedAE_dimred_mean=("MedAE_dimred", "mean"),
           MedAE_dimred_median=("MedAE_dimred", "median"),
           MedAE_surface_dimred_mean=("MedAE_surface_dimred", "mean"),
           MedAE_surface_dimred_median=("MedAE_surface_dimred", "median"),
       ).reset_index()
   )

   summary.to_csv("simulation_summary.csv", index=False)




