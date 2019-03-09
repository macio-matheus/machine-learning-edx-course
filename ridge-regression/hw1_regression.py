import numpy as np
import sys

# lambda_input = int(sys.argv[1])
# sigma2_input = float(sys.argv[2])
# X_train = np.genfromtxt(sys.argv[3], delimiter=",")
# y_train = np.genfromtxt(sys.argv[4])
# X_test = np.genfromtxt(sys.argv[5], delimiter=",")


lambda_input = 2
sigma2_input = 3
X_train = np.loadtxt('X_train.csv', delimiter=",")
Y_train = np.loadtxt('y_train.csv', delimiter=",")
X_test = np.loadtxt('X_test.csv', delimiter=",")


def part1(x, y, lbd):
    """
    Solution part 1: Compute the ridge regression weights

    np.linalg.inv:

    Calcula o inverso de uma matriz.
    O inverso de uma matriz eh tal que, se for multiplicado pela matriz original,
    resulta em matriz de identidade.

    np.dot:
    Calcula o produto de duas matrizes

    np.eye:
    Calcula a matriz identidade

    :param lbd: lambda input
    :param x: train set
    :param y: labels train set
    :return: wRR, list with ridge regression wights
    """
    _, d = x.shape

    # inputs x identity matrix from x_train[1] + scalar product form x train matrix
    l_identity = lbd * np.eye(d)
    wrr = np.linalg.inv(l_identity + x.T.dot(x)).dot(x.T).dot(y)
    return wrr


def part2(x_train, x_test, lmbd, sigma):
    """
    Solution for Part 2, implemented  Active Learning
    :return: active,  points chosen using active learning
    """
    _, x = x_train.shape

    x_test_index = list(range(x_test.shape[0]))

    indexes = []

    # Calculate covariance
    covariance = np.linalg.inv(lmbd * np.eye(x) + 1 / sigma * (x_train.T.dot(x_train)))
    print("Cov matrix", covariance.shape)

    for _ in range(10):
        temp_cov = [x_test[i].dot(covariance).dot(x_test[i].T) for i in x_test_index]
        print("preds", temp_cov[:5])
        print("argmax", np.argmax(temp_cov[:5]))

        pos_max_variance = np.argmax(temp_cov)
        el = x_test_index.pop(pos_max_variance)
        indexes.append(el)

        # update covariance with test values
        covariance = np.linalg.inv(covariance +
                                   sigma * x_test[pos_max_variance].T.dot(x_test[pos_max_variance]))

        last_index = indexes[len(indexes) - 1]

        print("x_test shape:", x_test[last_index].shape)
        print("x_train shape:", x_train.shape)
        x_train = np.concatenate((x_train, x_test[last_index].reshape(1, x)), axis=0)

    return (np.array(indexes) + 1).reshape(1, len(indexes))


if __name__ == '__main__':
    wRR = part1(X_train, Y_train, lambda_input)  # Assuming wRR is returned from the function
    np.savetxt("wRR_" + str(lambda_input) + ".csv", wRR, delimiter="\n")  # write output to file

    active = part2(X_train, X_test, lambda_input,
                   sigma2_input)  # Assuming active is returned from the function
    print("Active", active)
    np.savetxt("active_" + str(lambda_input) + "_" + str(int(sigma2_input)) + ".csv", active,
               delimiter=",")  # write output to file
