import numpy as np
import sys

X_train = np.genfromtxt(sys.argv[1], delimiter=",")
k = 5
n = 10


def f(x, mu, sigma):
    mu1 = np.linalg.inv(sigma)
    return np.linalg.det(mu1) * np.exp(-1 / 2 * (x - mu).T.dot(mu1).dot(x - mu))


def KMeans(X):
    """
    k-means clustering is a method of vector quantization,
    originally from signal processing, that is popular for cluster analysis in data mining.

    k-means clustering aims to partition n observations into k clusters in which each observation
    belongs to the cluster with the nearest mean, serving as a prototype of the cluster.
    :param X: Train set
    """
    N = X.shape[1]
    centroids = np.random.random((5, N))
    for i in range(n):
        index = np.array([np.argmin(np.sum((centroids - x) ** 2, axis=1)) for x in X])
        for j in range(k):
            cnt = np.sum(index == j)
            if cnt > 0:
                centroids[j] = np.mean(X[index == j], axis=0)
        filename = "centroids-" + str(i + 1) + ".csv"  # "i" would be each iteration
        np.savetxt(filename, centroids, delimiter=",")


def EMGMM(X):
    """
    Expectation-Maximization algorithm for Gaussian mixtures, is an iterative algorithm that starts from some initial
    estimate of Θ (e.g., random), and then proceeds to iteratively update Θ until convergence is detected.
    :param X: Train set
    """

    N = X.shape[1]
    mu = np.random.random((k, N))
    sigma = np.array([np.eye(N) for _ in range(k)])
    pi = np.random.uniform(size=k)
    pi = pi / np.sum(pi)

    for i in range(n):
        _phi = []
        for x in X:
            phi = pi * np.array([f(x, mu[t], sigma[t]) for t in range(k)])
            phi = phi / np.sum(phi)
            _phi.append(phi)
        _phi = np.array(_phi)
        nu = np.sum(_phi, axis=0)
        pi = nu / nu.sum()
        print(np.sum(pi), nu.shape)

        mu = _phi.T.dot(X)
        mu /= nu.reshape((-1, 1))

        for j in range(k):
            xi = X - mu[j]
            diag = np.diag(_phi[:, j])
            s = xi.T.dot(diag).dot(xi)
            s /= nu[j]
            sigma[j] = s
        filename = "pi-" + str(i + 1) + ".csv"
        np.savetxt(filename, pi, delimiter=",")
        filename = "mu-" + str(i + 1) + ".csv"
        np.savetxt(filename, mu, delimiter=",")  # this must be done at every iteration

        for j in range(k):  # k is the number of clusters
            filename = "Sigma-" + str(j + 1) + "-" + str(
                i + 1) + ".csv"  # this must be done 5 times (or the number of clusters) for each iteration
            np.savetxt(filename, sigma[j], delimiter=",")


def main():
    KMeans(X_train)
    EMGMM(X_train)


if __name__ == '__main__':
    main()
