import matplotlib.pyplot as plt
import numpy as np
img = plt.imread("data/img/n35w107.jpg")
plt.imshow(img, extent=[-31.0,31.0,-31.0,31.0])
#path = np.load("data/npy/n35w107_poam.npy")
path =np.load("results/trainingv2n35w107/trajectories/episode_002_trajectory.npy")-31.0
print(path.shape)
plt.scatter(path[:,0], path[:,1])

plt.show()
