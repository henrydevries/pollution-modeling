import numpy as np
from basisfunctions import gammasequencemaker

def onestepforecaster(Gamma_train, Gamma_test, Phi, Phi_surface):
    train_days, K = Gamma_train.shape
    Gamma_forecast = np.zeros((1000, K))
    for k in range(K):
        history = list(Gamma_train[:, k])
        for i in range(1000):
            gammahat = np.mean(history[-20:])
            Gamma_forecast[i, k] = gammahat
            history.append(Gamma_test[i,k])

    return Gamma_forecast @ Phi.T, Gamma_forecast @ Phi_surface.T

def rollingaverage(Phi, Phi_surface, Y_list_train, Y_list_test, Mu_list_test, Mu_list_surface, R):
    Gamma_train = gammasequencemaker(Y_list_train, Phi, R)
    Gamma_test = gammasequencemaker(Y_list_test, Phi, R)

    Mutilde_list_test, Mutilde_list_surface = onestepforecaster(Gamma_train, Gamma_test, Phi, Phi_surface)

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





