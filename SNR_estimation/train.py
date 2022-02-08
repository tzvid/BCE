import os
import sys

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
device = 'cpu' #choose CPU or CUDA, 

## parameters

experiment_name = 'release'

arg1 = sys.argv[1] if len(sys.argv) > 1 else 1
arg2 = sys.argv[2] if len(sys.argv) > 2 else 0
lamda_mse = float(arg1)
lamda_bias = float(arg2)

seed = 0
test_seed = 0


n = 50
d = 1


SNR_min_train = 2
SNR_max_train = 50
SNR_min_val = 2
SNR_max_val = 50
max_s = 10 #maximal value of the mean


N = 100
N_val = 1000
val_size = 100
grid_val_size = 10
batch_size = 10

hidden_size = 50


iters = 300000
gamma = 0.1
milestones = [40000]
lr = 0.00002

log_interval = 1000

os.makedirs(f'runs/{experiment_name}', exist_ok=True)
writer = SummaryWriter(f'runs/{experiment_name}/{lamda_mse}_{lamda_bias}')

print((lamda_mse, lamda_bias))

def un_normalize_randn(x, min_SNR, max_SNR):
    x = x * (max_SNR - min_SNR) / 2 + (min_SNR + max_SNR) / 2
    return x


def generate_data(N,  min_SNR, max_SNR, batch_size=None, Ys = None, seed=None, only_SNR_change = False,):
    if seed is not None:
        np.random.seed(seed)

    if Ys is None:
        unif = (torch.rand(batch_size).float().to(device) - 0.5) * 2
        SNR = un_normalize_randn(unif, min_SNR, max_SNR)
    else:
      batch_size = len(Ys)
      SNR = torch.from_numpy(SNRs).float().to(device)

    SNR_reshape = SNR.reshape(-1, 1, 1)
    if only_SNR_change:
        s = (torch.ones([batch_size, 1, 1]) * torch.rand(1) * max_s).float().to(device)
        sigma = s / torch.sqrt(SNR_reshape)
        alpha = torch.tile(torch.from_numpy(np.random.choice([-1, 1], (1, N, n))).float().to(device), [batch_size, 1, 1])
        noise = sigma * torch.tile(torch.randn(1, N, n).float(), [batch_size, 1, 1]).to(device)

    else:
        s = ((torch.rand(batch_size, 1, 1) + 0.1) * max_s).float().to(device)
        sigma = s / torch.sqrt(SNR_reshape)
        alpha = torch.from_numpy(np.random.choice([-1, 1], (batch_size, N, n))).float().to(device)
        noise = sigma * torch.randn(batch_size, N, n).float().to(device)
    X = s * alpha + noise
    X = X / torch.sqrt(torch.mean(X**2, 2).reshape(batch_size, N, 1))
    Y = SNR_reshape
    return X, Y



class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.fc1 = nn.Linear(6, hidden_size, bias=True)
      self.fc2 = nn.Linear(hidden_size, 1, bias=True)      # self.fc1.weight.data = 0 *  self.fc1.weight.data + 1
    def forward(self, x):
        m2 = 1
        m4 = torch.mean(x**4, 1).reshape(-1, 1)
        m4_inv = 1 / m4
        m4_sqrt = torch.sqrt(m4)
        m4_sqrt_inv = torch.sqrt(m4_inv)
        m6 = torch.mean(x**6, 1).reshape(-1, 1)
        m1 = torch.mean(torch.abs(x), 1).reshape(-1, 1)
        est = 0.5 * torch.sqrt(torch.abs(6 * m2 ** 2 - 2 * m4)) / (m2 - 0.5 * torch.sqrt(torch.abs(6 * m2 ** 2 - 2 * m4)))
        x = torch.cat([m4, m6, m1, m4_sqrt, m4_sqrt_inv, est], dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def MLE_EM(X):
    test_size = X.shape[0]
    N_test = X.shape[1]
    a_ml = torch.sign(X)
    s_hat = torch.mean(a_ml * X, 2)
    sigma_hat = torch.mean((X - s_hat.reshape(test_size, N_test, 1) * a_ml) ** 2, 2)
    for i in range(10):
        a_ml = torch.tanh(X * s_hat.reshape(test_size, -1, 1) / sigma_hat.reshape(test_size, -1, 1))
        s_hat = torch.mean(a_ml * X, 2)
        sigma_hat = torch.mean((X**2) - s_hat.reshape(test_size, N_test, 1) ** 2, 2)

    s2_hat = s_hat ** 2
    Y_hat = s2_hat / sigma_hat * (n - 3) / n - 1 / n
    Y_hat = Y_hat.reshape(test_size, N_test, 1)
    return Y_hat

model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

np.random.seed(seed)
torch.random.manual_seed(seed)


X_test, Y_test = generate_data(N_val, SNR_min_val, SNR_max_val, batch_size=val_size)
X_test_flatten = X_test.reshape(N_val * val_size, n)

SNRs = np.linspace(SNR_min_val, SNR_max_val, grid_val_size)

X_test_grid, Y_test_grid = generate_data(N_val, SNR_min_val, SNR_max_val, Ys=SNRs, seed=test_seed, only_SNR_change=True)
X_test_flatten_grid = X_test_grid.reshape(N_val * grid_val_size, n)


Y_hat_MLE_grid = MLE_EM(X_test_grid)

biases_MLE = torch.mean(Y_test_grid - Y_hat_MLE_grid, 1)

mses_MLE = torch.sqrt(torch.mean((Y_test_grid - Y_hat_MLE_grid) ** 2, 1))

print(mses_MLE.T)
print(biases_MLE.T)



Y_hat_test_MLE = MLE_EM(X_test)
mse_losses = torch.mean((Y_hat_test_MLE - Y_test) ** 2, 1)
mse_loss = torch.mean(mse_losses)
bias_losses = torch.mean((Y_hat_test_MLE - Y_test), 1) ** 2
bias_loss = torch.mean(bias_losses)
loss = lamda_mse * mse_loss + lamda_bias * bias_loss

print('MLE', loss.item(), mse_loss.item(), bias_loss.item())



for it in range(iters):
    optimizer.zero_grad()
    X, Y = generate_data(N, SNR_min_train, SNR_max_train, batch_size=batch_size)
    X_flatten = X.reshape(N * batch_size, n)
    Y_hat_flatten = model(X_flatten)
    Y_hat = Y_hat_flatten.reshape(batch_size,  N, 1)
    mse_losses = torch.mean((Y_hat - Y) ** 2, 1)
    mse_loss = torch.mean(mse_losses)
    bias_losses = torch.mean((Y_hat - Y), 1) ** 2
    bias_loss = torch.mean(bias_losses)
    loss = lamda_mse * mse_loss + lamda_bias * bias_loss
    loss.backward()
    optimizer.step()
    scheduler.step()
    if it % log_interval == 0:
        Y_hat_flatten_test = model(X_test_flatten)
        Y_hat_test = Y_hat_flatten_test.reshape(val_size,  N_val, 1)
        mse_losses = torch.mean((Y_hat_test - Y_test) ** 2, 1)
        mse_loss_test = torch.mean(mse_losses)
        bias_losses = torch.mean((Y_hat_test - Y_test), 1) ** 2
        bias_loss_test = torch.mean(bias_losses)
        loss_test = lamda_mse * mse_loss_test + lamda_bias * bias_loss_test
        print(it, loss.item(), loss_test.item(), mse_loss_test.item(), bias_loss_test.item())
        writer.add_scalar('train/loss', loss, it)
        writer.add_scalar('test/loss', loss_test, it)
        writer.add_scalar('test/bias_loss', bias_loss_test, it)
        writer.add_scalar('test/mse_loss', mse_loss_test, it)

    if it % (log_interval * 10) == 0:
        Y_hat_flatten_test_grid = model(X_test_flatten_grid)
        Y_hat_test_grid = Y_hat_flatten_test_grid.reshape(grid_val_size, N_val, 1)

        biases = torch.mean(Y_test_grid - Y_hat_test_grid, 1)

        mses = torch.sqrt(torch.mean((Y_test_grid - Y_hat_test_grid) ** 2, 1))

        print(mses.T)
        print(biases.T)
        writer.add_scalar('grid_test/mse_min', mses[0], it)
        writer.add_scalar('grid_test/mse_max', mses[-1], it)

    if it % (log_interval * 10) == 0:
        model_path = f'models_SNR/{experiment_name}/{lamda_mse}_{lamda_bias}/{it}'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
