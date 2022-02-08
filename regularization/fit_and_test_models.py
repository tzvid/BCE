import os

import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
np.random.seed(0)
n = 20
d = 20



num_repeats = 500
sigma = 1
sqrtS1 = np.random.randn(n, n) / (n)
Sigma1 = sqrtS1.T @ sqrtS1 * sigma**2

sqrtSy = np.random.randn(d, d) / (d)

sigma_y = 10
Sigma_y_diag = np.eye(d)

for i in range(5):
    Sigma_y_diag[i, i] = 0.01 / sigma_y ** 2

Sigma_y = Sigma_y = sqrtSy.T @ Sigma_y_diag @ sqrtSy * sigma_y ** 2

sqrtSy2 = np.random.randn(2 * d, d) / (2 * d)
Sigma_y2 = sqrtSy2.T @ sqrtSy2

H = np.random.randn(n, d)

def generate_sample_y_squared(Sigma_y, N):
    x = np.random.multivariate_normal(mean=np.zeros(d), cov=Sigma_y, size=(N))
    sample_y_sqaured = x.T @ x / N
    return sample_y_sqaured

def calc_estimator(sample_y_sqaured, lamda):
    lamda = lamda +1
    A = np.linalg.inv(H.T @ np.linalg.inv(Sigma1) @ H + 1 / lamda * np.linalg.inv(sample_y_sqaured) ) @ H.T @ np.linalg.inv(Sigma1)
    return A


def calc_ridge_estimator(sample_y_sqaured, lamda):
    A = np.linalg.inv(H.T @ np.linalg.inv(Sigma1+lamda * np.eye(n)) @ H + np.linalg.inv(sample_y_sqaured) ) @ H.T @ (np.linalg.inv(Sigma1+lamda * np.eye(n)))
    return A
def calc_MSE(A, Sigma_y):
    bias_squared = np.trace((A @ H - np.eye(d)) @ Sigma_y @ (A @ H - np.eye(d)).T)
    variance = np.trace(A @ Sigma1 @ A.T)
    mse = bias_squared + variance
    return mse, bias_squared, variance


Ns = np.arange(20, 80, 1)
lamdas = np.linspace(0, 10, 100)
fig = plt.figure()

BCE_mses = []
MMSE_mses = []

print ('running. this can take a while')

for N in Ns:
    mse = np.zeros(len(lamdas))
    np.random.seed(0)
    for i in range(num_repeats):
        sample_y_sqaured = generate_sample_y_squared(Sigma_y, N)
        for j, lamda in enumerate(lamdas):
            A = calc_estimator(sample_y_sqaured, lamda)
            mse_trial, bias_squared, variance = calc_MSE(A, Sigma_y)
            mse[j] += mse_trial / num_repeats
    best_mse = np.min(mse)
    MMSE_mse = mse[0]
    BCE_mses.append(best_mse)
    MMSE_mses.append(MMSE_mse)
    print(N)

lamdas = np.linspace(-0.012, 0.002, 100)

Ridge_mses = []
for N in Ns:
    mse = np.zeros(len(lamdas))
    np.random.seed(0)
    for i in range(num_repeats):
        sample_y_sqaured = generate_sample_y_squared(Sigma_y, N)
        for j, lamda in enumerate(lamdas):
            A = calc_ridge_estimator(sample_y_sqaured, lamda)
            mse_trial, bias_squared, variance = calc_MSE(A, Sigma_y)
            mse[j] += mse_trial / num_repeats

    best_mse = np.min(mse)
    Ridge_mses.append(best_mse)
    print(N)

fig, (ax1) = plt.subplots(1, 1, figsize=(5, 4), constrained_layout=True)

ax1.plot(Ns, BCE_mses, label='BCE')
ax1.plot(Ns, MMSE_mses, label='EMMSE')
ax1.plot(Ns, Ridge_mses, label='Ridge')
ax1.set_xlabel('# of samples')
ax1.set_ylabel('MSE')
ax1.legend()
plt.savefig(f'plots/linear/BCE_vs_RIDGE_hard_samples.png')
plt.show()




