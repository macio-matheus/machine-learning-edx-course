import numpy as np

users = 300
objects = 500

data = np.empty([0, 3])

for i in range(users):
    obs = np.random.choice(objects, size=np.random.randint(1, 100), replace=False)
    for j in obs:
        data = np.append(data, np.array([[i, j, np.random.randint(1, 6)]]), axis=0)

np.savetxt('ratings.csv', data, delimiter=',')