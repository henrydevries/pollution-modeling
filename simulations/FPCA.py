import numpy as np
from basisfunctions import Phimaker


def meanfunction(Y_list, Phi, R, l=0.01):
    K = Phi.shape[1]
    Gamma = np.zeros((len(Y_list), K))
    for i, y in enumerate(Y_list):
        mask = ~np.isnan(y)
        y_obs = y[mask]
        Phi_obs = Phi[mask]
        Gamma[i] = np.linalg.solve(Phi_obs.T @ Phi_obs + l * R, Phi_obs.T @ y_obs) # numerically more stable than regular inverse

    return Gamma @ Phi.T, Gamma

def globalmeanfunction(Y_list, Phi, Phi_surface, R, l=0.01):
    K = Phi.shape[1]

    A = np.zeros((K,K))
    b = np.zeros(K)

    for y in Y_list:
        mask = ~np.isnan(y)
        y_obs = y[mask]
        Phi_obs = Phi[mask]

        A += Phi_obs.T @ Phi_obs
        b += Phi_obs.T @ y_obs

    gamma = np.linalg.solve(A + l* R, b)
    return Phi @ gamma, Phi_surface @ gamma


def eigensolver(Kappa_centered, Omega0):
    T, K = Kappa_centered.shape

    eigvals0, eigvecs0 = np.linalg.eigh(Omega0)
    Omega0_sqrt = eigvecs0 @ np.diag(np.sqrt(eigvals0)) @ eigvecs0.T
    Omega0_sqrt_inv = eigvecs0 @ np.diag(1/np.sqrt(eigvals0)) @ eigvecs0.T

    W_sqrt = np.kron(Omega0_sqrt, Omega0_sqrt)
    W_sqrt_inv = np.kron(Omega0_sqrt_inv, Omega0_sqrt_inv)

    Sigma_transform = W_sqrt @ (Kappa_centered.T @ Kappa_centered / T) @ W_sqrt

    eigvals, eigvecs = np.linalg.eigh(Sigma_transform)

    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    Q = W_sqrt_inv @ eigvecs
    return eigvals, Q

def reconstructor(eigvals, Q, C, Omega0, Phi, Muhat_list, r):
    W = np.kron(Omega0, Omega0)
    Phi_eig = Phi @ Q # columns eigenfunctions and rows locations
    Xi = C @ W @ Q
    X_reconstructed = Muhat_list + Xi[:, :r] @ Phi_eig[:, :r].T
    return X_reconstructed, Xi[:, :r], Phi_eig[:, :r]


def FPCA(Y_list, t1, t2, r, Phi_surface, Omega0, R):
    T, N = Y_list.shape
    Phi = Phimaker(t1, t2)

    Muhat, Muhat_surface = globalmeanfunction(Y_list, Phi, Phi_surface, R)
    Muhat_list = np.tile(Muhat, (T, 1))
    Y_demeaned_list = Y_list - Muhat_list
    _, Kappa = meanfunction(Y_demeaned_list, Phi, R)
    Kappa_centered = Kappa - np.mean(Kappa, axis=0, keepdims=True)

    eigvals, Q = eigensolver(Kappa_centered, Omega0)
    Phi_surface_eig = Phi_surface @ Q[:, :r]
    _, Xi_truncated, Phi_eig_truncated = reconstructor(eigvals, Q, Kappa_centered, Omega0, Phi, Muhat_list, r)

    return Xi_truncated, Phi_eig_truncated, Phi_surface_eig, Muhat, Muhat_surface