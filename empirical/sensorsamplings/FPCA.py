import numpy as np
from scipy.interpolate import BSpline

def CBSPL(x, k, n_interior_knots, create_constant: bool = False):
    knots = np.concatenate(([0] * k, np.linspace(0, 1, n_interior_knots), [1] * k))
    basis_functions = [BSpline(knots, np.eye(len(knots) - k - 1)[i], k) for i in range(len(knots) - k - 1)]

    basis_values = np.array([b(x) for b in basis_functions])
    basis_values_d1 = [b.derivative(1)(x) for b in basis_functions]
    basis_values_d2 = [b.derivative(2)(x) for b in basis_functions]

    if create_constant:
        const = np.ones((1, x.size))
        zero = np.zeros((1, x.size))

        basis_values = np.vstack([const, basis_values])
        basis_values_d1 = np.vstack([zero, basis_values_d1])
        basis_values_d2 = np.vstack([zero, basis_values_d2])

    return basis_values, basis_values_d1, basis_values_d2

def omegacomputer(n_grid=2000, k=3, n_knots=5):
    grid  = np.linspace(0, 1, n_grid)

    basis_values, basis_values_d1, basis_values_d2 = CBSPL(grid, k, n_knots)

    K = basis_values.shape[0]

    Omega0 = np.zeros((K, K))
    Omega1 = np.zeros((K, K))
    Omega2 = np.zeros((K, K))

    for i in range(K):
        for j in range(K):
            Omega0[i, j] = np.trapz(basis_values[i] * basis_values[j], grid)
            Omega1[i, j] = np.trapz(basis_values_d1[i] * basis_values_d1[j], grid)
            Omega2[i, j] = np.trapz(basis_values_d2[i] * basis_values_d2[j], grid)
                        
    Omega0 = (Omega0 + Omega0.T)/2
    Omega1 = (Omega1 + Omega1.T)/2
    Omega2 = (Omega2 + Omega2.T)/2
    return Omega0, Omega1, Omega2

def penaltymatrixmaker(Omega0, Omega1, Omega2):
    R =  np.kron(Omega0, Omega2) + 2 * np.kron(Omega1, Omega1) + np.kron(Omega2, Omega0)
    return R

def basismatrixcomputer():
    Omega0, Omega1, Omega2 = omegacomputer()
    R = penaltymatrixmaker(Omega0, Omega1, Omega2)
    return Omega0, R

def Phimaker(s1, s2, k=3, n_knots=5):
    B1, _, _  = CBSPL(s1, k, n_knots)
    B2, _, _ = CBSPL(s2, k, n_knots)

    N = s1.shape[0]
    K1, K2 = B1.shape[0], B2.shape[0]
    Phi = np.empty((N, K1 * K2))

    for i in range(N):
        phi = np.kron(B2[:, i], B1[:, i])
        Phi[i,:] = phi.flatten()

    return Phi

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