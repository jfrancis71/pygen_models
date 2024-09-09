import argparse
import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pyro.nn
import pygen.train.train as train
import pygen.train.callbacks as callbacks
import pygen_models.distributions.made as made


class Made(nn.Module):
    def __init__(self, net, num_vars, made_params):
        super().__init__()
        self.made = made.MadeBernoulli(net, num_vars, made_params)

    def log_prob(self, x):
        return self.made.log_prob(nn.Flatten()(x))

    def sample(self, sample_shape):
        sample = self.made.sample(sample_shape)
        return nn.Unflatten(-1, [1, 28, 28])(sample)


class CategoricalImageLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = pyro.nn.ConditionalAutoRegressiveNN(784, 10, [784*2, 784*2], param_dims=[1],
            permutation=torch.arange(784))

    def forward(self, x):
        return Made(self.net, 784, nn.functional.one_hot(x, num_classes=10))


parser = argparse.ArgumentParser(description='PyGen Conditional PixelCNN')
parser.add_argument("--datasets_folder", default="~/datasets")
parser.add_argument("--tb_folder", default=None)
parser.add_argument("--images_folder", default=None)
parser.add_argument("--device", default="cpu")
parser.add_argument("--max_epoch", default=10, type=int)
parser.add_argument("--dummy_run", action="store_true")
ns = parser.parse_args()

torch.set_default_device(ns.device)

transform = transforms.Compose([transforms.ToTensor(),
    lambda x: (x > 0.5).float(), train.DevicePlacement()])
dataset = datasets.MNIST(ns.datasets_folder, train=True, download=False, transform=transform)
event_shape = [1, 28, 28]
data_split = [55000, 5000]

categorical_image_layer = CategoricalImageLayer()

train_dataset, validation_dataset = random_split(dataset, data_split,
    generator=torch.Generator(device=torch.get_default_device()))
tb_writer = SummaryWriter(ns.tb_folder)
epoch_end_callbacks = [
    callbacks.tb_log_image(tb_writer, "conditional_generated_images",
        callbacks.demo_conditional_images(categorical_image_layer, torch.arange(10), num_samples=2)),
    callbacks.tb_epoch_log_metrics(tb_writer),
    callbacks.tb_dataset_metrics_logging(tb_writer, "validation", validation_dataset)]
if ns.images_folder is not None:
    epoch_end_callbacks.append(
        callbacks.file_log_image(ns.images_folder,"conditional_generated_images",
        callbacks.demo_conditional_images(categorical_image_layer, torch.arange(10), num_samples=2)))
train.train(
    categorical_image_layer, train_dataset, train.layer_objective(reverse_inputs=True),
    batch_end_callback=callbacks.tb_batch_log_metrics(tb_writer),
    epoch_end_callback=callbacks.callback_compose(epoch_end_callbacks), dummy_run=ns.dummy_run, max_epoch=ns.max_epoch)
