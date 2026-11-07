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
        gammahat = np.linalg.lstsq(Phistar, ystar, rcond=None)[0]
    else:
        gammahat = np.linalg.solve(Phistar.T @ Phistar + l * R, Phistar.T @ ystar)
    return gammahat

def onestepforecaster(Y_list_train, Y_list_test, Phi, Phi_surface, R, lags):
    N, K = Phi.shape
    Gamma_forecast = np.zeros((len(Y_list_test), K))
    history = list(Y_list_train[-lags:])
    for i in range(len(Y_list_test)):
        gammahat = pooledgammaestimation(history[-lags:], Phi, R)
        Gamma_forecast[i] = gammahat
        history.append(Y_list_test[i])

    return Gamma_forecast @ Phi.T, Gamma_forecast @ Phi_surface.T


def pooling(Phi, Phi_surface, Y_list_train, Y_list_test, Mu_list_test, Mu_list_surface, R):
    Mutilde_list_test, Mutilde_list_surface = onestepforecaster(Y_list_train, Y_list_test, Phi, Phi_surface, R, lags=20)
    errors = np.abs(
        Mu_list_test - Mutilde_list_test
    )
    MAE = np.nanmean(errors)
    MedAE = np.nanmedian(errors)

    errors_surface = np.abs(
        Mu_list_surface[-1000:] - Mutilde_list_surface
    )
    MAE_surface = np.nanmean(errors_surface)
    MedAE_surface = np.nanmedian(errors_surface)
    return MAE, MAE_surface, MedAE, MedAE_surface
