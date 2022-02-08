import os

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
device = 'cpu'


seed = 0
test_seed = 0
experiment_name = 'release'

N = 100
N_test = 100
test_size = 100
grid_test_size = 100
batch_size = 10
n = 50
d = 1
it1 = 200000
it2 = 200000


SNR_min_test = 2
SNR_max_test = 50




max_s = 10

hidden_size = 50


def un_normalize_randn(x):
    x = x * (SNR_min_test - SNR_min_test) / 2 + (SNR_min_test + SNR_min_test) / 2
    return x



def generate_data(N, batch_size=None, Ys = None, seed=None, only_SNR_change = False):
    if seed is not None:
        np.random.seed(seed)

    if Ys is None:
        unif = (torch.rand(batch_size).float().to(device) - 0.5) * 2
        SNR = un_normalize_randn(unif)
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
        s = ((torch.rand(batch_size, 1, 1) + 0.1)* max_s).float().to(device)
        sigma = s / torch.sqrt(SNR_reshape)
        alpha = torch.from_numpy(np.random.choice([-1, 1], (batch_size, N, n))).float().to(device)
        noise = sigma * torch.randn(batch_size, N,n).float().to(device)
    X = s * alpha + noise
    X = X / torch.sqrt(torch.mean(X**2, 2).reshape(batch_size, N, 1))
    Y = SNR_reshape
    return X, Y




class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.fc1 = nn.Linear(6, hidden_size, bias=True)
      self.fc2 = nn.Linear(hidden_size, 1, bias=True)

    def forward(self, x):
        m2 = 1
        m4 = torch.mean(x**4, 1).reshape(-1, 1)
        m4_inv = 1 / m4
        m4_sqrt = torch.sqrt(m4)
        m4_sqrt_inv = torch.sqrt(m4_inv)
        m6 = torch.mean(x ** 6, 1).reshape(-1, 1)
        m1 = torch.mean(torch.abs(x), 1).reshape(-1, 1)
        est = 0.5 * torch.sqrt(torch.abs(6 * m2 ** 2 - 2 * m4)) / (m2 - 0.5 * torch.sqrt(torch.abs(6 * m2 ** 2 - 2 * m4)))
        x = torch.cat([m4, m6, m1, m4_sqrt, m4_sqrt_inv, est], dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def MLE_EM(X, test_size):
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


np.random.seed(seed)
torch.random.manual_seed(seed)



SNRs = np.linspace(SNR_min_test, SNR_max_test, grid_test_size)


X_test_grid, Y_test_grid = generate_data(N_test, Ys=SNRs, seed=test_seed, only_SNR_change=True)
X_test_flatten_grid = X_test_grid.reshape(N_test * grid_test_size, n)

Y_hat_MLE_grid = MLE_EM(X_test_grid, grid_test_size)

biases_unbiased = torch.mean(Y_test_grid-Y_hat_MLE_grid, 1)

mses_unbiased = torch.sqrt(torch.mean((Y_test_grid-Y_hat_MLE_grid) ** 2, 1))
mses_unbiased_inverse = torch.sqrt(torch.mean((1 / Y_test_grid - 1 / Y_hat_MLE_grid) ** 2, 1))


lamda_mse = 0.001
lamda_bias = 1.0

model = Net().to(device)
model_path = f'models_SNR/{experiment_name}/{lamda_mse}_{lamda_bias}/{it1}'
model.load_state_dict(torch.load(model_path,  map_location=device))



Y_hat_flatten_test_grid = model(X_test_flatten_grid)
Y_hat_test_grid = Y_hat_flatten_test_grid.reshape(grid_test_size, N_test, 1)

biases = torch.mean(Y_test_grid - Y_hat_test_grid, 1)

mses = torch.sqrt(torch.mean((Y_test_grid - Y_hat_test_grid) ** 2, 1))
mses_inverse = torch.sqrt(torch.mean((1 / Y_test_grid - 1 / Y_hat_test_grid) ** 2, 1))

lamda_mse = 1.0
lamda_bias = 0.0
#
model2 = Net().to(device)
model_path = f'models_SNR/{experiment_name}/{lamda_mse}_{lamda_bias}/{it2}'
model2.load_state_dict(torch.load(model_path,  map_location=device))



Y_hat_flatten_test_grid = model2(X_test_flatten_grid)
Y_hat_test_grid2 = Y_hat_flatten_test_grid.reshape(grid_test_size, N_test, 1)

biases2 = torch.mean(Y_test_grid - Y_hat_test_grid2, 1)

mses2 = torch.sqrt(torch.mean((Y_test_grid - Y_hat_test_grid2) ** 2, 1))

mses2_inverse = torch.sqrt(torch.mean((1 / Y_test_grid - 1 / Y_hat_test_grid2) ** 2, 1))

SNR_db = 10.0 * np.log10(SNRs)
os.makedirs(f'plots/{experiment_name}', exist_ok=True)

print(torch.mean(mses ** 2).item())
print(torch.mean(mses2 ** 2).item())

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
ax1.semilogy(SNR_db, mses.data ** 2, label='BCE')
ax1.semilogy(SNR_db, mses_unbiased ** 2, label='MLE', color='green')
ax1.semilogy(SNR_db, mses2.data ** 2, label='EMMSE', color='orange')
ax1.set_xlabel('SNR [db]')
ax1.set_ylabel('MSE')
ax1.set_title('MSE of SNR estimation as a function of the SNR')
ax1.legend()

ax2.plot(SNR_db, biases.data, label='BCE')
ax2.plot(SNR_db, biases_unbiased, label='MLE', color='green')
ax2.plot(SNR_db, biases2.data, label='EMMSE', color='orange')
ax2.set_xlabel('SNR [db]')
ax2.set_ylabel('Bias')
ax2.set_title('Bias of SNR estimation as a function of the SNR')
ax2.legend()

ax3.semilogy(SNR_db, mses_inverse.data, label='BCE')
ax3.semilogy(SNR_db, mses_unbiased_inverse, label='MLE', color='green')
ax3.semilogy(SNR_db, mses2_inverse.data, label='EMMSE', color='orange')
ax3.set_xlabel('SNR [db]')
ax3.set_ylabel('MSE')
ax3.set_title('MSE of 1/SNR estimation as a function of the SNR')
ax3.legend()


plt.savefig(f'plots/{experiment_name}/bias_and_mse_and_inverse.png')
plt.show()



X_test, Y_test = generate_data(N_test, batch_size=test_size)
X_test_flatten = X_test.reshape(N_test * test_size, n)

Y_hat_flatten_test = model(X_test_flatten)
Y_hat_test = Y_hat_flatten_test.reshape(test_size, N_test, 1)
mse_losses = torch.mean((Y_hat_test - Y_test) ** 2, 1)
mse_loss_test = torch.mean(mse_losses)
bias_losses = torch.mean((Y_hat_test - Y_test), 1) ** 2
bias_loss_test = torch.mean(bias_losses)

print('BCE', mse_loss_test.item(), bias_loss_test.item())


Y_hat_test_MLE = MLE_EM(X_test, test_size)
mse_losses = torch.mean((Y_hat_test_MLE - Y_test) ** 2, 1)
mse_loss = torch.mean(mse_losses)
bias_losses = torch.mean((Y_hat_test_MLE - Y_test), 1) ** 2
bias_loss = torch.mean(bias_losses)
loss = lamda_mse * mse_loss + lamda_bias * bias_loss



