from scipy import stats
import numpy as np
from basisfunctions import Phimaker

def MarkovMaskMaker(n_loc, days, p, q):
    Z = np.zeros((n_loc, days), dtype=bool)
    pi1 = (1-q)/(2-p-q)
    Z[:, 0] = np.random.rand(n_loc) < pi1
    for t in range(1, days):
        prev = Z[:, t-1]
        stay_1 = (np.random.rand(n_loc) < p)
        stay_0 = (np.random.rand(n_loc) < q)
        Z[:, t] = np.where(prev, stay_1, ~stay_0)
    return Z


def factordynamics(T, sigma_u, r=5):
    Xi = np.zeros((T, r))
    c = 0.01 * np.ones(r)
    A = np.diag(np.linspace(0.1, 0.8, r))
    Xi[0] = np.linalg.inv((np.eye(r) - A)) @ c
    for t in range(T-1):
        Xi[t+1] = c + A @ Xi[t] + sigma_u * np.random.randn(r)
    return Xi

def MarkovMaskMaker(n_loc, days, p, q):
    '''
    p: P(Z_{t} = 1 | Z_{t-1} = 1)
    q: P(Z_{t} = 0 | Z_{t-1} = 0)
    '''
    Z = np.zeros((n_loc, days), dtype=bool)
    pi1 = (1-q)/(2-p-q)
    Z[:, 0] = np.random.rand(n_loc) < pi1
    for t in range(1, days):
        prev = Z[:, t-1]
        stay_1 = (np.random.rand(n_loc) < p)
        stay_0 = (np.random.rand(n_loc) < q)
        Z[:, t] = np.where(prev, stay_1, ~stay_0)
    return Z


def prepare_innovations(locations, h):
    diff = locations[:, None, :] - locations[None, :, :]
    distances = np.linalg.norm(diff, axis=2)
    Sigma = np.exp(-distances / h)
    return Sigma


def innovationmaker(observed_locations, Sigma, nu):
    eps = stats.multivariate_t.rvs(
        shape=Sigma,
        df=nu
    )
    return eps[observed_locations]

def DGPmaker(r, s1, s2, Phi, T, Z, nu, h=0.1,):
    N, K = Phi.shape
    sigma_u = 1.0
    Xi = factordynamics(T, sigma_u, r)
    H = np.load("Q_empirical.npy")[:, :5]
    locations = np.column_stack((s1, s2))

    # 1. Generate all mean functions at once
    factor_scores = Xi @ H.T
    Mu_list = factor_scores @ Phi.T

    # 2. Precompute innovation quantities
    Sigma = prepare_innovations(locations, h)

    # 3. Generate innovations
    Y_list = np.full((T, N), np.nan)
    for t in range(T):
        observed_locations = np.flatnonzero(Z[:, t])

        if len(observed_locations):
            eps = innovationmaker(
                observed_locations,
                Sigma,
                nu=nu
            )

            Y_list[t, observed_locations] = (
                Mu_list[t, observed_locations] + eps
            )

    return Y_list, Mu_list

def surfacepicker(s1, s2, T, r, p, q, nu):
    s1_vals = np.linspace(0.05, 0.95, 10)
    s2_vals = np.linspace(0.05, 0.95, 10)

    S1, S2 = np.meshgrid(s1_vals, s2_vals)
    s1_surface = S1.ravel()
    s2_surface = S2.ravel()

    Phi_surface = Phimaker(s1_surface, s2_surface)
    Phi = Phimaker(s1, s2)

    Z = MarkovMaskMaker(len(s1), T, p=p, q=q)
    Z_surface = MarkovMaskMaker(len(s1_surface), T, p=1, q=0)

    Y_list, Mu_list = DGPmaker(
        r, s1, s2, Phi, T, Z, nu
    )

    _, Mu_list_surface = DGPmaker(
        r, s1_surface, s2_surface, Phi_surface,
        T, Z_surface, nu
    )

    return Phi, Phi_surface, Y_list, Mu_list, Mu_list_surface