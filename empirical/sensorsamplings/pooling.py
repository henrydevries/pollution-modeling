import numpy as np

def pooledgammaestimation(Y_list, Phi, R, l=0.01):
    ystar = []
    Phistar = []
    for y in Y_list:
        mask = ~np.isnan(y)
        y_valid = y[mask]
        ystar.append(y_valid)
        Phi_valid = Phi[mask]
        Phistar.append(Phi_valid)

    ystar = np.concatenate(ystar)
    Phistar = np.concatenate(Phistar)

    if Phi.shape[1] < 49:
        gammatilde = np.linalg.lstsq(Phistar, ystar, rcond=None)[0]
    else:
        gammatilde = np.linalg.solve(Phistar.T @ Phistar + l * R, Phistar.T @ ystar)
    return gammatilde

def regularCV(Y_list_train, Phi, R):
    L_candidates = np.array([0.0001, 0.001, 0.01, 0.1, 1])
    MAE = np.zeros_like(L_candidates)
    for i, l in enumerate(L_candidates):
        Mutilde_train_list, _ = onestepforecaster(Y_list_train[:20], Y_list_train[20:], Phi, Phi, R, l)
        MAE[i] = np.nanmean(np.abs(Y_list_train[20:] - Mutilde_train_list))

    return L_candidates[np.argmin(MAE)]

def onestepforecaster(Y_list_train, Y_list_test, Phi, Phi_surface, R, l, lags=20):
    N, K = Phi.shape
    Gamma_forecast = np.zeros((len(Y_list_test), K))
    history = list(Y_list_train[-lags:])
    for i in range(len(Y_list_test)):
        gammahat = pooledgammaestimation(history[-lags:], Phi, R, l)
        Gamma_forecast[i] = gammahat
        history.append(Y_list_test[i])

    return Gamma_forecast @ Phi.T, Gamma_forecast @ Phi_surface.T

def pooling(Phi, Phi_surface, Y_list_train, Y_list_surface_test, Y_list_test, Level_list, Level_list_eval, R):
    lstar = regularCV(Y_list_train, Phi, R)
    Mutilde_list_test, Mutilde_list_surface_test = onestepforecaster(Y_list_train, Y_list_test, Phi, Phi_surface, R, l=lstar)

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
