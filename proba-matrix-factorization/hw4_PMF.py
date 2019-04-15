from __future__ import division
import numpy as np
import sys

train_data = np.genfromtxt(sys.argv[1], delimiter=",")

lam = 2  # L2 regularization
mu = 0
sigma2 = 0.1
d = 5
iteration = 50


def upd_u(lam_param, sigma, d, M, m_missing, V, N1, N2):
    p1 = lam_param * sigma * np.eye(d)
    U = np.zeros((N1, d))

    for i in range(N1):
        p2 = np.zeros((d, d))
        p3 = np.zeros((d, 1))

        for j in range(N2):
            if not m_missing[i, j]:
                Vj = V[:, j].reshape(d, 1)
                p2 += np.dot(Vj, Vj.T)
                p3 += M[i, j] * Vj

        U[i, :] = np.dot(np.linalg.inv(p1 + p2), p3).reshape(-1)

    return U


def upd_v(lam_param, sigma, d, M, m_missing, u, n1, N2):
    p1 = lam_param * sigma * np.eye(d)
    V = np.zeros((d, N2))

    for j in range(N2):
        p2 = np.zeros((d, d))
        p3 = np.zeros((d, 1))

        for i in range(n1):
            if not m_missing[i, j]:
                Ui = u[i, :].reshape(d, 1)
                p2 += np.dot(Ui, Ui.T)
                p3 += M[i, j] * Ui

        V[:, j] = np.dot(np.linalg.inv(p1 + p2), p3).reshape(-1)

    return V


def calc_L(lam_param, sigma, m, m_missing, u, v, n1, n2):
    p = 0
    for i in range(n1):
        for j in range(n2):
            if not m_missing[i, j]:
                p += (m[i, j] - np.dot(u[i, :], v[:, j].T)) ** 2
    p /= 2 * sigma
    return - p - lam_param / 2 * (((np.linalg.norm(u, axis=1)) ** 2).sum()) - lam_param / 2 * (
        ((np.linalg.norm(v, axis=0)) ** 2).sum())


def PMF(x_train):
    """
    Matrix factorization is trying to predict the values of a matrix using 2 lower rank matrices.

    The total number of parameters to train is N * M * K . (N = number of users, M = number of items, K = latent dimension)

    :param x_train: ratings.csv A rating is a triple (user, item, rating)
    """
    # initialize matrix
    n1, n2 = int(np.amax(x_train[:, 0])), int(np.amax(x_train[:, 1]))
    L, u_matrices, v_matrices = np.zeros((iteration, 1)), np.zeros((iteration, n1, d)), np.zeros((iteration, d, n2))

    # generate v randomly
    V = np.random.normal(mu, 1 / lam, (d, n2))

    # build M matrix
    M = np.zeros((n1, n2))
    m_missing = np.ones((n1, n2), dtype=np.int32)
    for rating in x_train:
        r = int(rating[0])
        c = int(rating[1])
        M[r - 1, c - 1] = rating[2]
        m_missing[r - 1, c - 1] = 0

    for i in range(iteration):
        v_matrices[i] = V

        # update U
        U = upd_u(lam, sigma2, d, M, m_missing, V, n1, n2)
        u_matrices[i] = U

        # calculate L
        L[i] = calc_L(lam, sigma2, M, m_missing, U, V, n1, n2)

        # update V
        V = upd_v(lam, sigma2, d, M, m_missing, U, n1, n2)

    return L, u_matrices, v_matrices


def main():
    # Assuming the PMF function returns Loss L, U_matrices and V_matrices (refer to lecture)
    L, U_matrices, V_matrices = PMF(train_data)

    np.savetxt("objective.csv", L, delimiter=",")

    np.savetxt("U-10.csv", U_matrices[9], delimiter=",")
    np.savetxt("U-25.csv", U_matrices[24], delimiter=",")
    np.savetxt("U-50.csv", U_matrices[49], delimiter=",")

    np.savetxt("V-10.csv", V_matrices[9].T, delimiter=",")
    np.savetxt("V-25.csv", V_matrices[24].T, delimiter=",")
    np.savetxt("V-50.csv", V_matrices[49].T, delimiter=",")


if __name__ == '__main__':
    main()
