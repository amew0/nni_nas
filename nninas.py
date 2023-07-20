# %
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn.functional as F

import nni
import nni.nas.nn.pytorch as nn_
import nni.nas.nn.pytorch._layers as nn

# %
# class DepthwiseSeparableConv(torch.nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, groups=in_ch)
#         self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

#     def forward(self, x):
#         return self.pointwise(self.depthwise(x))


class MyModelSpace(nn_.ModelSpace):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # LayerChoice is used to select a layer between Conv2d and DwConv.
        self.conv2 = nn_.LayerChoice([
            nn.Conv2d(32, 64, 3, 1),
            nn.Conv2d(32, 64, 3, 2)
            # DepthwiseSeparableConv(32, 64)
        ],label='conv2')
        self.dropout1 = nn.Dropout(nni.choice(label='dropout1',choices=[0.25, 0.5, 0.75]))  # choose dropout rate from 0.25, 0.5 and 0.75
        self.dropout2 = nn.Dropout(0.5)
        feature = nni.choice(label='feature',choices=[64, 128, 256])
        self.fc1 = nn.Linear(9216, feature)
        self.fc2 = nn.Linear(feature, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(self.conv2(x), 2)
        x = torch.flatten(self.dropout1(x), 1)
        x = self.fc2(self.dropout2(F.relu(self.fc1(x))))
        output = F.log_softmax(x, dim=1)
        return output


model_space = MyModelSpace()


# %
import nni.nas.strategy as strategy
import nni.nas.evaluator.pytorch.lightning as pl

from torchvision import transforms
from torchvision.datasets import MNIST
# transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
# train_dataset = MNIST('data/mnist', download=True, transform=transf)
# test_dataset = MNIST('data/mnist', download=True, train=False, transform=transf)

# # %
# evaluator = pl.Classification(
#   # Need to use `pl.DataLoader` instead of `torch.utils.data.DataLoader` here,
#   # or use `nni.trace` to wrap `torch.utils.data.DataLoader`.
  
#   train_dataloaders=pl.DataLoader(train_dataset, batch_size=100),
#   val_dataloaders=pl.DataLoader(test_dataset, batch_size=100),
#   # Other keyword arguments passed to pytorch_lightning.Trainer.
#   max_epochs=10,
# )

def train_epoch(model, device, train_loader, optimizer, epoch):
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test_epoch(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
          correct, len(test_loader.dataset), accuracy))

    return accuracy

def evaluate_model(model):
    # By v3.0, the model will be instantiated by default.
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = pl.DataLoader(MNIST('data/mnist', download=True, transform=transf), batch_size=100),
    test_loader = pl.DataLoader(MNIST('data/mnist', download=True, train=False, transform=transf), batch_size=100),

    for epoch in range(3):
        # train the model for one epoch
        train_epoch(model, device, train_loader, optimizer, epoch)
        # test the model for one epoch
        accuracy = test_epoch(model, device, test_loader)
        # call report intermediate result. Result can be float or dict
        nni.report_intermediate_result(accuracy)

    # report final test result
    nni.report_final_result(accuracy)

from nni.nas.evaluator import FunctionalEvaluator
evaluator = FunctionalEvaluator(evaluate_model)

# %
# exploration_strategy = strategy.DARTS()
exploration_strategy = strategy.Random()

# %
from nni.nas.experiment import NasExperiment
exp = NasExperiment(model_space, evaluator, exploration_strategy)

exp.config.max_trial_number = 3   # spawn 3 trials at most
exp.config.trial_concurrency = 1  # will run 1 trial concurrently
exp.config.trial_gpu_number = 1
# exp.config.training_service.use_active_gpu = True

exp.run(8083)

