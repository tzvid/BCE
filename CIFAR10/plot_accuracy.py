import numpy as np

import matplotlib.pyplot as plt

epochs = [50, 60, 70, 80, 90]

bce = np.zeros([5, len(epochs)])
emmse = np.zeros([5, len(epochs)])
for i, epoch in enumerate(epochs):
    EMMSE_file = f'results/results_release_1_False_{epoch}.txt'

    BCE_file = f'results/results_release_True_1_{epoch}.txt'

    EMMSE = np.loadtxt(EMMSE_file, delimiter=',')
    BCE = np.loadtxt(BCE_file, delimiter=',')
    num_crops = BCE[:, 0]
    bce[:, i] = BCE[:, 1]
    emmse[:, i] = EMMSE[:, 1]

fig, (ax) = plt.subplots(1, 1, figsize=(5, 4), constrained_layout=True)


ax.errorbar(num_crops, np.mean(bce,1), np.std(bce,1), label='BCE')
ax.errorbar(num_crops, np.mean(emmse,1), np.std(emmse,1),  label='EMMSE')
ax.set_xlabel('number of test-time augmentation crops')
ax.set_ylabel('accuracy')
ax.set_xticks([1, 5, 10, 15, 20])
ax.legend()
plt.savefig('results/CIFAR10_with_std.png')
plt.show()
