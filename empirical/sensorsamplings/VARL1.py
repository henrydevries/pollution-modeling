import numpy as np
from statsmodels.tsa.api import VAR

def gammasequencemaker(Y_list, Phi, R, l=0.01):
    T, N = Y_list.shape
    K = Phi.shape[1]
    Gamma = np.zeros((T, K))
    for i, y in enumerate(Y_list):
        mask = ~np.isnan(y)
        y_obs = y[mask]
        Phi_obs = Phi[mask]
        if K==49:
            Gamma[i] = np.linalg.solve(Phi_obs.T @ Phi_obs + l * R, Phi_obs.T @ y_obs)
        elif K<49:
            Gamma[i] = np.linalg.lstsq(Phi_obs, y_obs, rcond=None)[0]
    return Gamma

def regularCV(Y_list_train, Phi, R):
    L_candidates = np.array([0.0001, 0.001, 0.01, 0.1, 1, 10])
    MAE = np.zeros_like(L_candidates)
    T = Y_list_train.shape[0]
    split = int(T * 0.7)
    for i, l in enumerate(L_candidates):
        Gamma_train = gammasequencemaker(Y_list_train[:split], Phi, R, l)
        Gamma_test = gammasequencemaker(Y_list_train[split:], Phi, R, l)
        Mutilde_test_list, _ = onestepforecaster(Gamma_train, Gamma_test, Phi, Phi)
        MAE[i] = np.nanmean(np.abs(Y_list_train[split:] - Mutilde_test_list))

    return L_candidates[np.argmin(MAE)]


def onestepforecaster(Gamma_train, Gamma_test, Phi, Phi_surface):
    T, K = Gamma_train.shape
    Gamma_forecast = np.zeros((1000, K))
    model = VAR(Gamma_train).fit(maxlags=1)
    history = Gamma_train.copy()
    for i in range(Gamma_test.shape[0]):
        Gamma_forecast[i] = model.forecast(y=history[-1:], steps=1)
        history = np.vstack([history, Gamma_test[i]])
    return Gamma_forecast @ Phi.T, Gamma_forecast @ Phi_surface.T

def VARL1(Phi, Phi_surface, Y_list_train, Y_list_surface_test, Y_list_test, Level_list, Level_list_eval, R):
    lstar = regularCV(Y_list_train, Phi, R)
    Gamma_train = gammasequencemaker(Y_list_train, Phi, R, l=lstar)
    Gamma_test = gammasequencemaker(Y_list_test, Phi, R, l=lstar)

    Mutilde_list_test, Mutilde_list_surface_test = onestepforecaster(Gamma_train, Gamma_test, Phi, Phi_surface)

    if Phi.shape[1] < 49:
        errors = np.abs(
            np.exp(Mutilde_list_test + Level_list[-365:])
            - np.exp(Y_list_test + Level_list[-365:])
        )
        MAE = np.nanmean(errors)
        MedAE = np.nanmedian(errors)

        errors_surface = np.abs(
            np.exp(Mutilde_list_surface_test + Level_list_eval[-365:])
            - np.exp(Y_list_surface_test + Level_list_eval[-365:])
        )
        MAE_surface = np.nanmean(errors_surface)
        MedAE_surface = np.nanmedian(errors_surface)

    elif Phi.shape[1] == 49:
        errors = np.abs(
            np.exp(Mutilde_list_test) - np.exp(Y_list_test)
        )
        MAE = np.nanmean(errors)
        MedAE = np.nanmedian(errors)

        errors_surface = np.abs(
            np.exp(Mutilde_list_surface_test) - np.exp(Y_list_surface_test)
        )
        MAE_surface = np.nanmean(errors_surface)
        MedAE_surface = np.nanmedian(errors_surface)
        
    return MAE, MAE_surface, MedAE, MedAE_surface
