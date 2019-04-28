from __future__ import division
import numpy as np
import sys

# X_train = np.genfromtxt(sys.argv[1], delimiter=",")
# Y_train = np.genfromtxt(sys.argv[2], delimiter=',')
# X_test = np.genfromtxt(sys.argv[3], delimiter=",")


X_train = np.loadtxt('X_train.csv', delimiter=",")
Y_train = np.loadtxt('y_train.csv', delimiter=",")
X_test = np.loadtxt('X_test.csv', delimiter=",")


def calc_parameters_likelihood_mle(_classes, _x_train, _y_train):
    """
    Calc likelihood MLE
    :param _classes: classes dataset
    :param _x_train: train set x
    :param _y_train: train label
    :return: mus, sigmas, sigma_dets
    """
    mus, sigmas, sigma_dets = {}, {}, {}
    for c in _classes:
        c_id = (_y_train == c)
        X = _x_train[c_id, :]
        mu = np.matrix(np.mean(X, axis=0)).T
        # calculate gaussian covariance mle
        n, _ = X.shape
        sigma = 0
        for i in range(n):
            x = np.matrix(X[i, :]).T
            sigma += (x - mu) * (x - mu).T / n
        sigma_det = np.linalg.det(sigma)
        mus[c], sigmas[c], sigma_dets[c] = mu, sigma, sigma_det

    return mus, sigmas, sigma_dets


def plugin_classifier(x_train, y_train, x_test):
    """
    Implements bayes classifier
    :param x_train:
    :param y_train:
    :param x_test:
    :return: prob_pred for test set
    """

    # calculate prior by MLE
    n = y_train.shape[0]
    classes = np.unique(y_train)

    priors = {}
    for c in classes:
        priors[c] = np.sum(y_train == c) / n

    # calculate parameters of likelihood by MLE
    mus, sigmas, sigma_dets = calc_parameters_likelihood_mle(classes, x_train, y_train)

    c = classes.shape[0]

    n, d = x_test.shape
    prob_pred = np.zeros((n, c))

    for i in range(n):
        x = np.matrix(x_test[i, :]).T

        i_prob = np.zeros(c)
        for j in range(c):
            # calculate Gaussian likelihood
            mu = mus[classes[j]]
            sigma = sigmas[classes[j]]
            sigma_det = sigma_dets[classes[j]]

            # calc multi gaussian prob and proportion of posterior
            i_prob[j] = priors[classes[j]] * np.sqrt(sigma_det) ** (-1) * np.exp(
                - 0.5 * (x - mu).T * sigma.I * (x - mu))

        prob_pred[i, :] = i_prob / np.sum(i_prob)

    return prob_pred


def main():
    # assuming final_outputs is returned from function
    final_outputs = plugin_classifier(X_train, Y_train, X_test)

    # write output to file
    np.savetxt("probs_test.csv", final_outputs, delimiter=",")


if __name__ == '__main__':
    main()