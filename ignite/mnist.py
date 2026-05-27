# This example is from https://github.com/pytorch/ignite/tree/master
# See above for license.

import argparse
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
import torchvision.datasets as datasets
from torchvision.transforms import Compose, Normalize, ToTensor
from ignite.engine import create_supervised_evaluator, create_supervised_trainer, Events
from ignite.metrics import Accuracy, Loss
from ignite.utils import setup_logger
from ignite.engine import Events, Engine
from pygen.train import callbacks
from pygen.neural_nets import classifier_net
import pygen.layers.independent_categorical as layer_categorical


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.vars = nn.Parameter(torch.ones([1, 28, 28], requires_grad=True))

    def log_prob(self, x):
        dist = torch.distributions.Bernoulli(logits=self.vars)
        return dist.log_prob(x)

    def sample(self):
        dist = torch.distributions.Bernoulli(logits=self.vars)
        return dist.sample()

mymodel = Model()


parser = argparse.ArgumentParser()
parser.add_argument("--datasets_folder", default="~/datasets")
parser.add_argument("--tb_folder", default=None)
parser.add_argument("--device", default="cpu")
parser.add_argument("--max_epoch", type=int, default=10, help="number of epochs to train (default: 10)")
args = parser.parse_args()

transform = Compose([ToTensor(), lambda x: (x > 0.5).float()])
dataset = datasets.MNIST(args.datasets_folder, train=True, download=True, transform=transform)
data_split = [55000, 5000]
train_dataset, validation_dataset = random_split(dataset, data_split)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
#model = classifier_net.ClassifierNet(mnist=True)
mymodel.to(args.device)  # Move model before creating optimizer
optimizer = SGD(mymodel.parameters(), lr=.001)
criterion = nn.CrossEntropyLoss()


# Training step function
def train_step(engine, batch):
    x, y = batch
    optimizer.zero_grad()
    #y_hat = mymodel.log_prob(torch.tensor(x, dtype=torch.int))
    y_hat = mymodel.log_prob(x)
    loss = -y_hat.sum()
    loss.backward()
    optimizer.step()
    return loss.item()

# Ignite trainer
trainer = Engine(train_step)

#trainer = create_supervised_trainer(model, optimizer, criterion, device=args.device)
trainer.logger = setup_logger("trainer")

val_metrics = {"nll": Loss(criterion)}
#evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=args.device)
#evaluator.logger = setup_logger("evaluator")
example_valid_images = next(iter(torch.utils.data.DataLoader(validation_dataset, batch_size=25)))[0].to(args.device)
tb_writer = SummaryWriter(args.tb_folder)

@trainer.on(Events.EPOCH_COMPLETED)
def log_results(engine):
#    evaluator.run(train_loader)
#    metrics = evaluator.state.metrics
#    tb_writer.add_scalar("training/avg_loss", metrics["nll"], engine.state.epoch)
#    evaluator.run(val_loader)
#    metrics = evaluator.state.metrics
#    tb_writer.add_scalar("validation/avg_loss", metrics["nll"], engine.state.epoch)
    if tb_writer is not None:
        image = mymodel.sample()
        tb_writer.add_image("valid_images", image, engine.state.epoch)

trainer.run(train_loader, max_epochs=args.max_epoch)
