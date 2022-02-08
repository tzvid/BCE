from __future__ import print_function
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import pickle as pkl

import numpy as np
import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader

experiment_name = 'release'

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(6272, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)

        output = F.log_softmax(x, dim=1)
        return output

    def soft_labels(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x



class soft_labels_dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_path, transform, pil=False, num_crops=20):

        with open(data_path, 'rb') as f:
            self.images, self.labels, self.soft_labels = pkl.load(f)
            self.images = [torch.from_numpy(self.images[i]) for i in range(len(self.images))]
            self.transform = transform
            self.pil = pil
            self.num_crops = num_crops

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = many_crops(self.images[idx], num_crops=self.num_crops)
        return self.transform(img), torch.tensor(self.labels[idx]), torch.tensor(self.soft_labels[idx])

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target, soft_target) in enumerate(train_loader):
        shape = data.shape
        target = torch.from_numpy(np.tile(target.reshape(shape[0], 1, 1).numpy(), [1,args.num_crops,1])).reshape(-1)
        data, target, soft_target = data.to(device), target.to(device), soft_target.to(device)
        data = data.reshape(-1, shape[2], shape[3], shape[4])
        optimizer.zero_grad()
        output = model.soft_labels(data)
        output_hard = F.log_softmax(output, dim=1)
        output_soft = F.log_softmax(output / args.T, dim=1).reshape(shape[0], args.num_crops, 10)
        soft_target = F.log_softmax(soft_target / args.T, dim=1)
        soft_target = soft_target.reshape(-1, 1, 10)
        if args.BCE:
            loss1 = torch.mean(torch.mean(output_soft - soft_target, 1) ** 2)
        else:
            loss1 = torch.mean((output_soft - soft_target) ** 2)

        loss2 = F.nll_loss(output_hard, target)
        loss = args.lamda1 * loss1 + args.lamda2 * loss2
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data) /args.num_crops, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader, res_file, topk=2, num_crops=20):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        i = 0
        acc1 = 0
        acck = 0
        for data, target, _ in test_loader:
            data, target = data.to(device), target.to(device)
            shape = data.shape
            data = data.reshape(shape[0] * shape[1], shape[2],shape[3],shape[4])
            output = model(data)
            output = output.reshape(shape[0], shape[1], 10)
            output = torch.mean(output, 1)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            a1, ak = accuracy(output,  target, topk=(1,topk))
            acc1 += a1
            acck += ak
            i+=1
        acc1 /= i
        acck /= i


    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print(num_crops)
    print(acc1.item())
    print(acck.item())
    print ('--------')

    with open (res_file, 'a') as f:
        f.write(f'{num_crops},{acc1.item()}, {acck.item()} \n')

def save_soft_labels(model, device, test_loader):
    model.eval()
    images = []
    soft_labels = []
    labels = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            soft_target = model(data)
            data_list = [d for d in data.data.cpu().numpy()]
            labels_list = [d for d in target.data.cpu().numpy()]
            soft_labels_list = [t for t in soft_target.data.cpu().numpy()]

            images += data_list
            labels += labels_list
            soft_labels += soft_labels_list
    os.makedirs('data', exist_ok=True)
    with open('data/cifar_soft_labels_full.pkl', 'wb') as f:
        pkl.dump([images, labels, soft_labels], f)


def many_crops(img, size=32, padding=4, num_crops=20):
        transform = transforms.Compose([transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=size, padding=padding)
                                        ])

        img = [transform(img) for i in range(num_crops)]
        return img


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10000, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--train-size', type=int, default=10000,
                        )
    parser.add_argument('--test-size', type=int, default=10000,
                        )
    parser.add_argument('--T', type=float, default=20,
                        )
    parser.add_argument('--lamda1', type=float, default=1,
                        )
    parser.add_argument('--lamda2', type=float, default=1,
                        )
    parser.add_argument('--BCE', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--student', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--load-epoch', type=int, default=0,
                        )
    parser.add_argument('--num-crops', type=int, default=20,
                        )
    parser.add_argument('--topk', type=int, default=2,
                        )
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform1 = transforms.Compose([
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(crop) for crop in crops])),
    ])



    path_test = 'data/cifar_soft_labels_test.pkl'

    dataset2 = soft_labels_dataset(path_test, transform=transform1,  num_crops=args.num_crops
                                   )


    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    os.makedirs('results', exist_ok=True)
    res_file = f"results/results_{experiment_name}_{args.lamda1}_{args.BCE}_{args.load_epoch}.txt"
    model = Net().to(device)

    if args.load_epoch:
        model_path = f"models/{experiment_name}_{args.lamda1}_{args.BCE}_epoch_{args.load_epoch}.pt"
        model.load_state_dict(torch.load(model_path,  map_location=device))


    test(model, device, test_loader, res_file, topk=args.topk, num_crops=args.num_crops)


if __name__ == '__main__':
    main()