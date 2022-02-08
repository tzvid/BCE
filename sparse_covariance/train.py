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

## parameters


experiment_name = 'release'

seed = 0
test_seed = 1
cov_seed = 0

n = 200
d = 5

indices = [(0, 3), (1, 2), (3, 4), (2, 4)]



N = 20
N_test = 20
test_size = 100
batch_size = 1


iters = 1000000
log_interval = 1000

min_y_train = 0.00
max_y_train = 1

min_y_test = 0.0
max_y_test = 1

lr = 0.0001

arg1 = sys.argv[1] if len(sys.argv) > 1 else 1
arg2 = sys.argv[2] if len(sys.argv) > 2 else 0
arg3 = sys.argv[3] if len(sys.argv) > 3 else -1

lamda_mse = float(arg1)
lamda_bias = float(arg2)

os.makedirs(f'runs/{experiment_name}', exist_ok=True)
writer = SummaryWriter(f'runs/{experiment_name}/{lamda_mse}_{lamda_bias}')


layers = 50
hidden_size = 100
v_size = 50


print((lamda_mse, lamda_bias))

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



def generate_data(N, batch_size, test = False, seed=None):
    if seed is not None:
        np.random.seed(seed)

    if test:
        alphas = np.random.uniform(low=min_y_test, high=max_y_test, size=(batch_size, num_cov))
    else:
        alphas = np.random.uniform(low=min_y_train, high=max_y_train, size=(batch_size, num_cov))
    cov = 0
    omegas_reshape = alphas.reshape(-1, num_cov, 1, 1, 1)
    for i in range(num_cov):
        cov += omegas_reshape[:, i] * covs[i]
    cov += np.eye(d)
    X = np.array([np.random.multivariate_normal(np.zeros(d), cov[i,0], (N, n)) for i in range(batch_size)])
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

    inverse_I_cov = covs_mat.T[np.newaxis,:,:] @ inverse_I @ covs_mat[np.newaxis,:,:]
    mse_bound = np.trace(inverse_I_cov, axis1=1, axis2=2) / d ** 2
    return mse_bound


np.random.seed(seed)
torch.random.manual_seed(seed)
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5, verbose=True)
np.random.seed(0)


X_test, Y_test, alpha_test = generate_data(N_test, test_size, seed=test_seed, test=True)


X_test_flatten = X_test.reshape(N_test * test_size, n, d)


mse_crb = CRB(alpha_test)
print('crb', np.mean(mse_crb), np.min(mse_crb), np.max(mse_crb))


last_iter = int(arg3)
if last_iter > 0:
    model_path = f'models_SNR/{experiment_name}/{lamda_mse}_{lamda_bias}/{last_iter}'
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(model_path)


for it in range(last_iter, iters):
    optimizer.zero_grad()
    X, Y, _ = generate_data(N, batch_size=batch_size)
    X_flatten = X.reshape(N * batch_size, n, d)
    Y_hat_flatten = model(X_flatten)
    Y_hat = Y_hat_flatten.reshape(batch_size,  N, d*d)
    mse_losses = torch.mean((Y_hat - Y) ** 2, 1)

    mse_loss = torch.mean(mse_losses)
    bias_losses = torch.mean((Y_hat - Y), 1) ** 2
    bias_loss = torch.mean(bias_losses)

    loss = lamda_mse * mse_loss + lamda_bias * bias_loss
    loss.backward()
    optimizer.step()

    if it % log_interval == 0:
        Y_hat_flatten_test = model(X_test_flatten)
        Y_hat_test = Y_hat_flatten_test.reshape(test_size,  N_test, d*d)
        mse_losses = torch.mean((Y_hat_test - Y_test) ** 2, 1)
        mse_loss_test = torch.mean(mse_losses)
        bias_losses = torch.mean((Y_hat_test - Y_test), 1) ** 2
        bias_loss_test = torch.mean(bias_losses)
        loss_test = lamda_mse * mse_loss_test + lamda_bias * bias_loss_test
        min_mse = torch.min(torch.mean(mse_losses, 1)).item()
        max_mse = torch.max(torch.mean(mse_losses, 1)).item()
        print(it, loss.item(), loss_test.item(), mse_loss_test.item(), bias_loss_test.item(), min_mse, max_mse)
        writer.add_scalar('train/loss', loss, it)
        writer.add_scalar('test/loss', loss_test, it)
        writer.add_scalar('test/bias_loss', bias_loss_test, it)
        writer.add_scalar('test/mse_loss', mse_loss_test, it)
        scheduler.step(loss_test)

        if np.isnan(loss.item()):
            model = Net()
            model.load_state_dict(torch.load(model_path,  map_location=device))
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5,
                                                                   verbose=True)
            print (f'lodaded model from {model_path}')



    if it % (log_interval * 10) == 0:
        model_path = f'models_SNR/{experiment_name}/{lamda_mse}_{lamda_bias}/{it}'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)


