import argparse
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as datasets
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.utils import make_grid
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.utils import setup_logger
from pygen.train import callbacks


# https://github.com/hugobb/discreteVAE was useful for a working discrete VAE
# using Reinforce.

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 64)
        self.fc3 = nn.Linear(64, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x.view(-1, 784)))
        logits = self.fc2(h1)
        return logits

    def decode(self, z):
        h3 = F.relu(self.fc3(z.view(len(z), 64)))
        return self.fc4(h3).view(-1, 1, 28, 28)

    def log_prob(self, x):
        logits = self.encode(x)
        z_dist = torch.distributions.bernoulli.Bernoulli(logits=logits)
        z = z_dist.sample()
        log_prob_z = z_dist.log_prob(z)
        recon_logits = self.decode(z)
        dist = torch.distributions.Bernoulli(logits=recon_logits)
        log_prob_recon = dist.log_prob(x)
        log_prob_z = log_prob_z.mean(-1)
        reinforce_loss = log_prob_recon.sum(axis=[1,2,3]).detach() * log_prob_z
        recons_log_prob = log_prob_recon.sum(axis=[1,2,3])
        return recons_log_prob.mean(-1) + reinforce_loss.mean(-1) - reinforce_loss.detach().mean(-1), z_dist.entropy().mean(axis=1).mean(-1)

    def sample(self, x):
        latent = torch.distributions.Bernoulli(logits=self.encode(x)).sample()
        logits = self.decode(latent)
        dist = torch.distributions.Bernoulli(logits=logits)
        return dist.probs


def train_step(engine, batch):
    x, y = batch
    optimizer.zero_grad()
    loss = -mymodel.log_prob(x.to(args.device))[0]
    loss.backward()
    optimizer.step()

def evaluate_function(engine, batch):
    x, y = batch
    log_prob, z_entropy = mymodel.log_prob(x.to(args.device))
    return -log_prob, z_entropy


parser = argparse.ArgumentParser()
parser.add_argument("--datasets_folder", default="~/datasets")
parser.add_argument("--tb_folder", default=None)
parser.add_argument("--device", default="cpu")
parser.add_argument("--max_epoch", type=int, default=10, help="number of epochs to train (default: 10)")
args = parser.parse_args()

mymodel = Model()
mymodel.to(args.device)
transform = Compose([ToTensor(), lambda x: (x > 0.5).float()])
dataset = datasets.MNIST(args.datasets_folder, train=True, download=True, transform=transform)
data_split = [55000, 5000]
train_dataset, validation_dataset = random_split(dataset, data_split)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
optimizer = Adam(mymodel.parameters(), lr=.001)

trainer = Engine(train_step)
trainer.logger = setup_logger("trainer")
example_valid_images = next(iter(torch.utils.data.DataLoader(validation_dataset, batch_size=25)))[0].to(args.device)
tb_writer = SummaryWriter(args.tb_folder)
evaluator = Engine(evaluate_function)
evaluator.logger = setup_logger("evaluator")
metric = RunningAverage(output_transform=lambda x: x[0].item())
metric.attach(evaluator, "log_prob")
metric = RunningAverage(output_transform=lambda x: x[1].item())
metric.attach(evaluator, "z_entropy")


@trainer.on(Events.EPOCH_COMPLETED)
def log_results(engine):
    if tb_writer is not None:
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        tb_writer.add_scalar("validation/log_prob", metrics["log_prob"], engine.state.epoch)
        tb_writer.add_scalar("validation/z_entropy", metrics["z_entropy"], engine.state.epoch)
        images = mymodel.sample(example_valid_images)
        images = torch.cat([example_valid_images, images], dim=0)
        image = make_grid(images, nrow=25, padding=10)
        tb_writer.add_image("valid_images", image, engine.state.epoch)

trainer.run(train_loader, max_epochs=args.max_epoch)
