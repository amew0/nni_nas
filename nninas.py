# %
import torch
import torch.nn.functional as F
import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# %
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


@model_wrapper
class ModelSpace(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # LayerChoice is used to select a layer between Conv2d and DwConv.
        self.conv2 = nn.LayerChoice([
            nn.Conv2d(32, 64, 3, 1),
            DepthwiseSeparableConv(32, 64)
        ])
        # ValueChoice is used to select a dropout rate.
        # ValueChoice can be used as parameter of modules wrapped in `nni.retiarii.nn.pytorch`
        # or customized modules wrapped with `@basic_unit`.
        self.dropout1 = nn.Dropout(nn.ValueChoice([0.25, 0.5, 0.75]))  # choose dropout rate from 0.25, 0.5 and 0.75
        self.dropout2 = nn.Dropout(0.5)
        feature = nn.ValueChoice([64, 128, 256])
        self.fc1 = nn.Linear(9216, feature)
        self.fc2 = nn.Linear(feature, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(self.conv2(x), 2)
        x = torch.flatten(self.dropout1(x), 1)
        x = self.fc2(self.dropout2(F.relu(self.fc1(x))))
        output = F.log_softmax(x, dim=1)
        return output


model_space = ModelSpace()
model_space

# %
import nni.nas.strategy as strategy
import nni.nas.evaluator.pytorch.lightning as pl

from torchvision import transforms
from torchvision.datasets import MNIST
transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = MNIST('data/mnist', download=True, transform=transf)
test_dataset = MNIST('data/mnist', download=True, train=False, transform=transf)

# %
evaluator = pl.Classification(
  # Need to use `pl.DataLoader` instead of `torch.utils.data.DataLoader` here,
  # or use `nni.trace` to wrap `torch.utils.data.DataLoader`.
  
  train_dataloaders=pl.DataLoader(train_dataset, batch_size=100),
  val_dataloaders=pl.DataLoader(test_dataset, batch_size=100),
  # Other keyword arguments passed to pytorch_lightning.Trainer.
  max_epochs=10,
  gpus=1,
)


# %
exploration_strategy = strategy.DARTS()


