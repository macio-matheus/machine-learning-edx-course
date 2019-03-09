import numpy as np
from matplotlib import pyplot as plt


def db_gen():
    """
    Generate synthetic database for regression 
    :return:
    """
    x_random = np.random.random((100, 1)) * 5
    print(x_random.shape)
    x_train = np.hstack((x_random, np.ones_like(x_random)))
    print(x_train.shape)
    y_train = x_train.dot([[0.4], [3]])
    print(y_train.shape)
    y_train += (np.random.random(y_train.shape) - 0.5) * 0.3

    # Plot database
    plt.scatter(x_train[:, 0], y_train)
    plt.show()

    # Save file
    np.savetxt('x_train.csv', x_train, delimiter=',')
    np.savetxt('y_train.csv', y_train)
    np.savetxt('X_test.csv', x_train, delimiter=',')


if __name__ == '__main__':
    db_gen()
