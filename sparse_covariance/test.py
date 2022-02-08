import os
import sys

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from scipy.signal import find_peaks, savgol_filter

device = 'cpu'

it1 = 700000
it2 = 700000



seed = 0
test_seed = 1
cov_seed = 0

experiment_name = 'release'
model_dir = 'models_SNR'

os.makedirs(f'plots/{experiment_name}', exist_ok=True)

N_test = 500
test_size = 100
n = 200
d = 5


min_y_test = 0
max_y_test = 0.4




layers = 50

hidden_size = 100
v_size = 50

indices = [(0, 3), (1, 2), (3, 4), (2, 4)]


np.random.seed(cov_seed)
num_cov = d + len(indices)
covs = []
covs_torch = []
sigma = np.ones(num_cov) * 1


for i in range(d):
    cov = np.zeros([d, d])
    cov[i, i] = sigma[i] ** 2
    covs.append(cov)
    covs_torch.append(torch.from_numpy(covs[i]).float().to(device).reshape(d*d))



for i in range(num_cov-d):
    cov = np.zeros([d, d])
    cov[indices[i][0], indices[i][1]] = 0.5
    cov[indices[i][1], indices[i][0]] = 0.5
    covs.append(cov)
    covs_torch.append(torch.from_numpy(covs[-1]).float().to(device).reshape(d * d))

def generate_data(N, batch_size, grid = False, seed=None, min_y_train=0, max_y_train=1):
    if seed is not None:
        np.random.seed(seed)

    if grid:
        alphas = np.tile(np.linspace(min_y_test, max_y_test, batch_size).reshape(-1, 1), [1,num_cov])
    else:
        alphas = np.random.uniform(low=min_y_train, high=max_y_train, size=(batch_size, num_cov))
    cov = 0
    alphas_reshape = alphas.reshape(-1, num_cov, 1, 1, 1)
    for i in range(num_cov):
        cov += alphas_reshape[:, i] * covs[i]
    cov += np.eye(d)
    X = np.array([np.random.multivariate_normal(np.zeros(d), cov[i, 0], (N, n)) for i in range(batch_size)])
    Y = cov.reshape(batch_size, 1, d*d)
    X = torch.from_numpy(X).float().to(device)
    Y = torch.from_numpy(Y).float().to(device)
    return X, Y, alphas





class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = []
        self.fc2v = []
        self.fc2a = []

        for i in range(layers):
          self.fc1.append(nn.Linear(d*d+v_size, hidden_size, bias=True))
          self.fc2v.append(nn.Linear(hidden_size, v_size, bias=True))
          self.fc2a.append(nn.Linear(hidden_size, num_cov, bias=True))

        self.fc1 = nn.ModuleList(self.fc1)
        self.fc2v = nn.ModuleList(self.fc2v)
        self.fc2a = nn.ModuleList(self.fc2a)

    def forward(self, x):
        shape = x.shape
        x = torch.swapaxes(x,2,1) @ x / n
        y = x.reshape(-1, d * d)
        alpha = torch.ones(shape[0], num_cov) / 2
        alpha = alpha.to(device)
        alpha = alpha.reshape(-1, num_cov, 1)
        v = torch.zeros(shape[0], v_size).to(device)
        for i in range(layers):
            inp = torch.cat([v, y], 1)
            h = F.relu(self.fc1[i](inp))
            dv = self.fc2v[i](h)
            dalpha = self.fc2a[i](h)
            alpha = alpha + 0.1 * dalpha.reshape(-1, num_cov, 1)
            v = v + 0.1 * dv
            y = 0
            for j in range(num_cov):
                y += alpha[:, j] * covs_torch[j]
            y += torch.eye(d).reshape(1,d,d).reshape(-1,d*d)
        return y


    def batch_forward(self, x, batch_size=1000):
        y = torch.zeros(x.shape[0], d*d)
        for i in range(int(x.shape[0] / batch_size)):
            y[i*batch_size:(i+1)*batch_size] = self.forward(x[i*batch_size:(i+1)*batch_size])
        return y






def CRB(alpha):
    batch_size = len(alpha)
    alpha = alpha.reshape(-1, num_cov, 1, 1, 1)
    cov = 0
    for i in range(num_cov):
        cov += alpha[:, i] * covs[i]
    cov += np.eye(d)
    inv_cov = np.linalg.inv(cov)
    I = np.zeros([batch_size, num_cov, num_cov])
    for i in range(num_cov):
        for j in range(num_cov):
            I[:, i, j] = 0.5 * np.squeeze(np.trace(inv_cov @ covs[i] @ inv_cov @ covs[j], axis1=2, axis2=3)) * n
    inverse_I = np.linalg.inv(I)
    covs_mat = np.array([c.reshape(d*d) for c in covs])

    inverse_I_cov = covs_mat.T[np.newaxis, :, :] @ inverse_I @ covs_mat[np.newaxis, :, :]
    mse_bound = np.trace(inverse_I_cov, axis1=1, axis2=2) / d ** 2
    return mse_bound


fig, (ax1) = plt.subplots(1, 1, figsize=(5, 3.5), constrained_layout=True)


np.random.seed(seed)
torch.random.manual_seed(seed)
model = Net().to(device)

np.random.seed(0)


X_test, Y_test, alpha_test = generate_data(N_test, test_size, seed=test_seed)


X_test_flatten = X_test.reshape(N_test * test_size, n, d)







mse_crb = CRB(alpha_test)


print('crb', np.mean(mse_crb), np.min(mse_crb), np.max(mse_crb))



lamda_mse = 0.001
lamda_bias = 1.0

model = Net().to(device)
model_path = f'{model_dir}/{experiment_name}/{lamda_mse}_{lamda_bias}/{it1}'
model.load_state_dict(torch.load(model_path,  map_location=device))

Y_hat_flatten_test = model.batch_forward(X_test_flatten)
Y_hat_test = Y_hat_flatten_test.reshape(test_size, N_test, d * d)
mse_losses = torch.mean(torch.mean((Y_hat_test - Y_test) ** 2, 1),1)
mse_loss_test = torch.mean(mse_losses)
bias_losses = torch.mean(torch.mean((Y_hat_test - Y_test), 1) ** 2, 1)
bias_loss_test = torch.mean(bias_losses)
loss = lamda_mse * mse_loss_test + lamda_bias * bias_loss_test
print('BCE', mse_loss_test.item(), torch.min(mse_losses).item(), torch.max(mse_losses).item(), loss.item(), bias_loss_test.item())




lamda_mse = 1.0
lamda_bias = 0.0

model = Net().to(device)
model_path = f'{model_dir}/{experiment_name}/{lamda_mse}_{lamda_bias}/{it2}'
model.load_state_dict(torch.load(model_path,  map_location=device))

Y_hat_flatten_test = model.batch_forward(X_test_flatten)
Y_hat_test = Y_hat_flatten_test.reshape(test_size, N_test, d * d)
mse_losses2 = torch.mean(torch.mean((Y_hat_test - Y_test) ** 2, 1),1)
mse_loss_test = torch.mean(mse_losses2)
bias_losses2 = torch.mean(torch.mean((Y_hat_test - Y_test), 1) ** 2, 1)
bias_loss_test = torch.mean(bias_losses2)
loss = lamda_mse * mse_loss_test + lamda_bias * bias_loss_test
print('EMMSE', mse_loss_test.item(), torch.min(mse_losses2).item(), torch.max(mse_losses2).item(), loss.item(), bias_loss_test.item())



ind = np.argsort(mse_crb)
ax1.semilogy(mse_losses[ind].data, '.', label='BCE')
ax1.semilogy(mse_losses2[ind].data, '.' ,label='EMMSE')
ax1.semilogy(mse_crb[ind], '.', label='crb')
ax1.legend(loc=4)

ax1.set_ylabel('MSE')
ax1.set_title('(a)', y=0.95, pad=-12, x=0.05, size=16)
ax1.set_xticks([])
plt.show()

#######################

np.random.seed(seed)
torch.random.manual_seed(seed)
model = Net().to(device)


X_test, Y_test, alpha_test = generate_data(N_test, test_size, seed=test_seed, min_y_train=0, max_y_train=0.4)


X_test_flatten = X_test.reshape(N_test * test_size, n, d)







mse_crb = CRB(alpha_test)


print('crb', np.mean(mse_crb), np.min(mse_crb), np.max(mse_crb))


#
lamda_mse = 0.001
lamda_bias = 1.0

model = Net().to(device)
model_path = f'{model_dir}/{experiment_name}/{lamda_mse}_{lamda_bias}/{it1}'
model.load_state_dict(torch.load(model_path,  map_location=device))

Y_hat_flatten_test = model.batch_forward(X_test_flatten)
Y_hat_test = Y_hat_flatten_test.reshape(test_size, N_test, d * d)
mse_losses = torch.mean(torch.mean((Y_hat_test - Y_test) ** 2, 1),1)
mse_loss_test = torch.mean(mse_losses)
bias_losses = torch.mean(torch.mean((Y_hat_test - Y_test), 1) ** 2, 1)
bias_loss_test = torch.mean(bias_losses)
loss = lamda_mse * mse_loss_test + lamda_bias * bias_loss_test
print('BCE', mse_loss_test.item(), torch.min(mse_losses).item(), torch.max(mse_losses).item(), loss.item(), bias_loss_test.item())



################################################3


lamda_mse = 1.0
lamda_bias = 0.0

model = Net().to(device)
model_path = f'{model_dir}/{experiment_name}/{lamda_mse}_{lamda_bias}/{it2}'
model.load_state_dict(torch.load(model_path,  map_location=device))

Y_hat_flatten_test = model.batch_forward(X_test_flatten)
Y_hat_test = Y_hat_flatten_test.reshape(test_size, N_test, d * d)
mse_losses2 = torch.mean(torch.mean((Y_hat_test - Y_test) ** 2, 1),1)
mse_loss_test = torch.mean(mse_losses2)
bias_losses2 = torch.mean(torch.mean((Y_hat_test - Y_test), 1) ** 2, 1)

bias_loss_test = torch.mean(bias_losses2)
loss = lamda_mse * mse_loss_test + lamda_bias * bias_loss_test
print('EMMSE', mse_loss_test.item(), torch.min(mse_losses2).item(), torch.max(mse_losses2).item(), loss.item(), bias_loss_test.item())


fig, (ax2) = plt.subplots(1, 1, figsize=(5, 3.5), constrained_layout=True)


ind = np.argsort(mse_crb)
ax2.semilogy(mse_losses[ind].data, '.', label='BCE')
ax2.semilogy(mse_losses2[ind].data, '.', label='EMMSE')
ax2.semilogy(mse_crb[ind], '.', label='crb')
ax2.legend(loc=4)

ax2.set_xticks([])
ax2.set_ylabel('MSE')

ax2.set_title('(b)', y=0.95, pad=-12, x=0.05, size=16)


plt.show()
