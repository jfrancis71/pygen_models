import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
import torchvision
from torchvision.utils import make_grid
import pyro.nn
from pygen.train import callbacks
import pygen.train.train as train
import pygen_models.train.train as pygen_models_train
import pygen_models.train.callbacks as pygen_models_callbacks
import pygen_models.distributions.made as made


class Made(nn.Module):
    def __init__(self):
        super().__init__()
        num_vars = 784
        net = pyro.nn.AutoRegressiveNN(num_vars, [num_vars*2, num_vars*2], param_dims=[1],
            permutation=torch.arange(num_vars))
        self.made = made.MadeBernoulli(net, 784, None)

    def log_prob(self, x):
        return self.made.log_prob(nn.Flatten()(x))

    def sample(self, sample_shape):
        sample = self.made.sample(sample_shape)
        return nn.Unflatten(1, [1, 28, 28])(sample)


parser = argparse.ArgumentParser(description='PyGen MNIST Discrete VAE')
parser.add_argument("--datasets_folder", default="~/datasets")
parser.add_argument("--tb_folder", default=None)
parser.add_argument("--images_folder", default=None)
parser.add_argument("--device", default="cpu")
parser.add_argument("--max_epoch", default=10, type=int)
parser.add_argument("--dummy_run", action="store_true")
ns = parser.parse_args()

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(), lambda x: (x > 0.5).float(),
    train.DevicePlacement()])
mnist_dataset = torchvision.datasets.MNIST(
    ns.datasets_folder, train=True, download=False,
    transform=transform)
train_dataset, validation_dataset = random_split(mnist_dataset, [50000, 10000])
torch.set_default_device(ns.device)

digit_distribution = Made()

tb_writer = SummaryWriter(ns.tb_folder)
epoch_end_callbacks_list = [
    callbacks.tb_epoch_log_metrics(tb_writer),
    callbacks.log_image_cb(pygen_models_callbacks.sample_images(digit_distribution),
            tb_writer=tb_writer, folder=ns.images_folder, name="conditional_generated_images"),
    callbacks.tb_dataset_metrics_logging(tb_writer, "validation", validation_dataset)]
epoch_end_callbacks = callbacks.callback_compose(epoch_end_callbacks_list)

train.train(digit_distribution, train_dataset, pygen_models_train.distribution_objective,
    batch_end_callback=callbacks.tb_batch_log_metrics(tb_writer),
    epoch_end_callback=epoch_end_callbacks, dummy_run=ns.dummy_run, max_epoch=ns.max_epoch, epoch_regularizer=False)
