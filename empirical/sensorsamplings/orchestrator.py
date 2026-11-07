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
from dataload import dataloader
from FPCA import FPCA, Phimaker, omegacomputer, penaltymatrixmaker
import rollingaverage
import statespace
import fGAS
import pooling
import VARL1
from joblib import Parallel, delayed
from dataclasses import dataclass
from itertools import product
import pandas as pd


def select_evaluation_sensors(N, n_eval):
    eval_idx = np.random.choice(N, size=n_eval, replace=False)
    train_idx = np.setdiff1d(range(N), eval_idx)
    return eval_idx, train_idx

def rescaler(x, x_min, x_max, eps=0.01):
    return np.clip(eps + (1 - 2 * eps) * (x - x_min) / (x_max - x_min), 0, 1)

def empiricalrun(spec, longitude, latitude, pollution, R, train_idx, eval_idx):
    lon_min, lon_max = longitude[train_idx].min(), longitude[train_idx].max()
    lat_min, lat_max = latitude[train_idx].min(), latitude[train_idx].max()
    s1 = rescaler(longitude[train_idx], lon_min, lon_max)
    s2 = rescaler(latitude[train_idx], lat_min, lat_max)
    s1_eval = rescaler(longitude[eval_idx], lon_min, lon_max)
    s2_eval = rescaler(latitude[eval_idx], lat_min, lat_max)
    Phi = Phimaker(s1, s2)
    Phi_surface = Phimaker(s1_eval, s2_eval)

    Y_list_surface = np.log(pollution[:, eval_idx])
    Y_list = np.log(pollution[:, train_idx])
    Y_list_train = Y_list[:-365]
    Y_list_test = Y_list[-365:]
    Y_list_surface_test = Y_list_surface[-365:]

    if "FPCA" in spec.basis:
        T, N = Y_list.shape
        if "3" in spec.basis:
            r = 3
        elif "7" in spec.basis:
            r = 7

        _, _, Phi_eig, Phi_surface_eig,  Muhat, Muhat_eval = FPCA(Y_list_train, Phi, Phi_surface, r)
        Level_list = np.tile(Muhat, (T,1))
        Level_list_eval = np.tile(Muhat_eval, (T,1))
        Y_list = Y_list - Level_list
        Y_list_surface = Y_list_surface - Level_list_eval
        Y_list_train = Y_list[:-365]
        Y_list_test = Y_list[-365:]
        Y_list_surface_test = Y_list_surface[-365:]

        MAE, MAE_surface, MedAE, MedAE_surface = spec.model(Phi_eig, Phi_surface_eig, Y_list_train, Y_list_surface_test, Y_list_test, Level_list, Level_list_eval, R)
    else:
        MAE, MAE_surface, MedAE, MedAE_surface = spec.model(Phi, Phi_surface, Y_list_train, Y_list_surface_test, Y_list_test, None, None, R)

    return {
        "basis": spec.basis,
        "model": spec.model_name,
        "MAE": MAE,
        "MAE_surface": MAE_surface,
        "MedAE" : MedAE, 
        "MedAE_surface" : MedAE_surface
    }
            

if __name__ == "__main__":
    # selecting sensor selection parameters
    M = 200

    # calculations that need to be done for each specification regardless of the sensors
    Omega0, Omega1, Omega2 = omegacomputer()
    R = penaltymatrixmaker(Omega0, Omega1, Omega2)
    _, longitude, latitude, pollution, _, _, _ = dataloader()

    @dataclass
    class Specification:
        basis: str
        model_name: str
        model: callable

    bases = [
        "FPCA(r=3)",
        "FPCA(r=7)",
        "Bspline"
    ]

    models = [
        ("fGAS", fGAS.fGAS),
        ("VAR(1)", VARL1.VARL1),
        ("State Space", statespace.statespace),
        ("Rolling Average", rollingaverage.rollingaverage),
        ("Pooling", pooling.pooling)
    ]

    specifications = [
        Specification(basis, model_name, model)
        for basis, (model_name, model) in product(bases, models)
    ]

    runs = []
    for iteration in range(M):
        eval_idx, train_idx = select_evaluation_sensors(
            N = len(longitude),
            n_eval = 75
        )
        for spec in specifications:
            runs.append(
                delayed(empiricalrun)(
                    spec,
                    longitude,
                    latitude,
                    pollution,
                    R,
                    train_idx,
                    eval_idx
                )
            )

    results = Parallel(n_jobs=192, verbose=10)(runs)
    results_df = pd.DataFrame(results)

    summary_df = (
        results_df
        .groupby(["basis", "model"])
        .agg(
            MAE_mean=("MAE", "mean"),
            MAE_median=("MAE", "median"),
            MAE_surface_mean=("MAE_surface", "mean"),
            MAE_surface_median=("MAE_surface", "median"),
            MedAE_mean =("MedAE", "mean"),
            MedAE_median = ("MedAE", "median"),
            MedAE_surface_mean = ("MedAE_surface", "mean"),
            MedAE_surface_median = ("MedAE_surface", "median")
        )
        .reset_index()
    )

    results_df.to_csv("results.csv", index=False)
    summary_df.to_csv("results_summary.csv", index=False)