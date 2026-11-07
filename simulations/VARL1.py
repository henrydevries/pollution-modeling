import numpy as np
from statsmodels.tsa.api import VAR
from basisfunctions import gammasequencemaker

def onestepforecaster(Gamma_train, Gamma_test, Phi, Phi_surface):
    T, K = Gamma_train.shape
    Gamma_forecast = np.zeros((1000, K))
    model = VAR(Gamma_train).fit(maxlags=1)
    history = Gamma_train.copy()
    for i in range(Gamma_test.shape[0]):
        Gamma_forecast[i] = model.forecast(y=history[-1:], steps=1)
        history = np.vstack([history, Gamma_test[i]])
    return Gamma_forecast @ Phi.T, Gamma_forecast @ Phi_surface.T

def VARL1(Phi, Phi_surface, Y_list_train, Y_list_test, Mu_list_test, Mu_list_surface, R):
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
