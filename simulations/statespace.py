import numpy as np

def Kalmanfilter(Y_list, Phi, B, omega, vareta, vareps):
    days, _ = Y_list.shape
    K = Phi.shape[1]

    g_pred = np.zeros((days, K))
    P_pred = np.zeros((days, K, K))
    g_filt = np.zeros((days, K))
    P_filt = np.zeros((days, K, K))

    g_pred[0] = np.zeros(K)
    P_pred[0] = np.eye(K)

    for t in range(days):
        y = Y_list[t]
        valid_mask = ~np.isnan(y)
        y_obs = y[valid_mask]
        Phi_obs = Phi[valid_mask]

        F_t  = Phi_obs @ P_pred[t] @ Phi_obs.T + vareps * np.eye(len(Phi_obs))
        v_t = y_obs - Phi_obs @ g_pred[t]
        K_t = P_pred[t] @ Phi_obs.T @ np.linalg.inv(F_t)
        g_filt[t] = g_pred[t] + K_t @ v_t
        P_filt[t] = P_pred[t] - K_t @ Phi_obs @ P_pred[t]

        if t < days - 1:
            g_pred[t+1] = B @ g_filt[t] + omega
            P_pred[t+1] = B @ P_filt[t] @ B.T + vareta * np.eye(K)

    return g_pred, P_pred, g_filt, P_filt

def Kalmansmoother(Y_list, Phi, g_pred, P_pred, vareps, vareta, B):
    days, _ = Y_list.shape
    K = Phi.shape[1]

    g_smooth = np.zeros_like(g_pred)
    P_smooth = np.zeros_like(P_pred)
    P_lag = np.zeros((days, K, K))


    r = np.zeros((days, K))
    N = np.zeros((days, K, K))

    r[-1] = 0
    N[-1] = 0

    for t in reversed(range(days)):
        y = Y_list[t]
        valid_mask = ~np.isnan(y)
        y_obs = y[valid_mask]
        Phi_obs = Phi[valid_mask]

        F_t  = Phi_obs @ P_pred[t] @ Phi_obs.T + vareps * np.eye(len(Phi_obs))
        K_t = B @ P_pred[t] @ Phi_obs.T @ np.linalg.inv(F_t)
        L_t = B - K_t @ Phi_obs
        v_t = y_obs - Phi_obs @ g_pred[t]

        if t > 0:
            r[t-1] = Phi_obs.T @ np.linalg.inv(F_t) @ v_t + L_t.T @ r[t]
            g_smooth[t] = g_pred[t] + P_pred[t] @ r[t-1]

            N[t-1] = Phi_obs.T @ np.linalg.inv(F_t) @ Phi_obs + L_t.T @ N[t] @ L_t
            P_smooth[t] = P_pred[t] - P_pred[t] @ N[t-1] @ P_pred[t]

        if t <  days - 1:
            P_lag[t] = P_pred[t] @ L_t.T @ (np.eye(K) - N[t] @ P_pred[t+1])
    return g_smooth, P_smooth, P_lag


def emUpdate(g_smooth, P_smooth, P_lag, Y_list, Phi):
    days, K = g_smooth.shape
    N, _ = Phi.shape

    ##### estimation of B and omega #######
    cross_moments = np.zeros((K, K + 1))
    auto_moments = np.zeros((K + 1, K + 1))

    for t in range(days-1):
        g_t = g_smooth[t]
        g_tp1 = g_smooth[t + 1]
        P_t = P_smooth[t]
        P_lag_t = P_lag[t]

        # E[z_t z_t^T]
        E_zt_ztT = np.block([
            [np.ones((1, 1)), g_t[None, :]],
            [g_t[:, None], P_t + np.outer(g_t, g_t)]
        ])

        # E[gamma_{t+1} z_t^T]
        E_gammatp1_ztT = np.hstack([
            g_tp1[:, None],
            P_lag_t.T + np.outer(g_tp1, g_t)
        ])

        cross_moments += E_gammatp1_ztT
        auto_moments += E_zt_ztT

    Theta_hat = cross_moments @ np.linalg.inv(auto_moments)  # (K, K+1)
    B_hat = Theta_hat[:, 1:]
    omega_hat = Theta_hat[:, 0]

    ##### estimation of eta #######
    sum_eta = 0
    for t in range(days-1):
        messy_var = P_smooth[t+1] + B_hat @ P_smooth[t] @ B_hat.T - P_lag[t].T @ B_hat.T - B_hat @ P_lag[t]
        etares = g_smooth[t+1] - (omega_hat + B_hat @ g_smooth[t])
        sum_eta += etares.T @ etares + np.trace(messy_var)

    vareta_hat = sum_eta / (K * (days -1 ))


    ##### estimation of epsilon #######
    sum_eps = 0
    for t in range(days):
        y = Y_list[t]
        valid_mask = ~np.isnan(y)
        y_obs = y[valid_mask]
        Phi_obs = Phi[valid_mask]

        epsres = y_obs - Phi_obs @ g_smooth[t]
        epsvar = Phi_obs @ P_smooth[t] @ Phi_obs.T
        sum_eps += epsres.T @ epsres + np.trace(epsvar)

    vareps_hat = sum_eps / (N * days)

    return B_hat, omega_hat, vareta_hat, vareps_hat

def em(Y_list, Phi, n_iter=10):
    K = Phi.shape[1]
    B_hat = np.eye(K) * 0.05
    omega_hat = np.zeros(K)
    vareta_hat = 1
    vareps_hat = 1

    for i in range(n_iter):
        g_pred, P_pred, _, _ = Kalmanfilter(Y_list, Phi, B_hat, omega_hat, vareta_hat, vareps_hat)
        g_smooth, P_smooth, P_lag = Kalmansmoother(Y_list, Phi, g_pred, P_pred, vareps_hat, vareta_hat, B_hat)
        B_hat, omega_hat, vareta_hat, vareps_hat = emUpdate(g_smooth, P_smooth, P_lag, Y_list, Phi)

    return B_hat, omega_hat, vareta_hat, vareps_hat

def onestepforecast(Y_list_train, Y_list_test, Phi, Phi_surface, B_hat, omega_hat, vareta_hat, vareps_hat):
    K = Phi.shape[1]
    Gamma_forecast = np.zeros((1000, K))
    history = Y_list_train
    for i in range(1000):
      _, _, g_filt, _ = Kalmanfilter(history[-100:], Phi, B_hat, omega_hat, vareta_hat, vareps_hat)
      Gamma_forecast[i] = B_hat @ g_filt[-1] + omega_hat
      history = np.vstack((history, Y_list_test[i:i+1]))

    return Gamma_forecast @ Phi.T, Gamma_forecast @ Phi_surface.T

def statespace(Phi, Phi_surface, Y_list_train, Y_list_test, Mu_list_test, Mu_list_surface, R):
    B_hat, omega_hat, vareta_hat, vareps_hat = em(Y_list_train, Phi, n_iter=10)
    Muhat_list_test, Muhat_list_surface = onestepforecast(Y_list_train, Y_list_test, Phi, Phi_surface, B_hat, omega_hat, vareta_hat, vareps_hat)

    errors = np.abs(
        Mu_list_test - Muhat_list_test
    )
    MAE = np.nanmean(errors)
    MedAE = np.nanmedian(errors)

    errors_surface = np.abs(
        Mu_list_surface[-1000:] - Muhat_list_surface
    )
    MAE_surface = np.nanmean(errors_surface)
    MedAE_surface = np.nanmedian(errors_surface)
    return MAE, MAE_surface, MedAE, MedAE_surface
