"""Simple Discrete VAE for training on MNIST.

Results default params, validation epoch: (Analytic -93.2, Uniform -111.3, ReinforceBaseline -94.25, Gumbel -95.77)
Results basic log_prob: 152, reconstruct: 133, kl_div: 17.6
Results simple pixelcn log_prob: 101, reconstruct: 96, kl_div: 5.2
"""


import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
import torchvision
from torchvision.utils import make_grid
from pygen.train import callbacks
from pygen.neural_nets import classifier_net
import pygen.train.train as train
import pygen.layers.independent_bernoulli as bernoulli_layer
import pygen_models.layers.independent_one_hot_categorical as one_hot_categorical
import pygen_models.layers.pixelcnn as pixelcnn_layer
import pygen_models.distributions.pixelcnn as pixelcnn_dist
import pygen_models.train.train as pygen_models_train


class VAE(nn.Module):
    def __init__(self, num_vars, num_states, p_x_given_z):
        super().__init__()
        self.num_states = num_states
        self.num_vars = num_vars
        self.p_x_given_z = p_x_given_z
        self.q_z_given_x = nn.Sequential(
            classifier_net.ClassifierNet(mnist=True, num_classes=self.num_states*self.num_vars),
            one_hot_categorical.IndependentOneHotCategorical([self.num_vars], self.num_states),
        )
        self.logits_p_z = torch.zeros([self.num_states])
        self.identity_matrix = torch.eye(num_states).float()

    def p_z(self):
        base_dist = torch.distributions.one_hot_categorical.OneHotCategorical(logits=torch.zeros([self.num_vars, self.num_states]))
        return torch.distributions.independent.Independent(base_distribution=base_dist, reinterpreted_batch_ndims=1)

    def kl_div(self, p, q):
        kl_div = torch.sum(p.base_dist.probs * (p.base_dist.logits - q.base_dist.logits), axis=-1)
        return kl_div.sum(axis=-1)

    def log_prob(self, x):
        return self.elbo(x)[0]

    def elbo(self, x):
        q_z_given_x = self.q_z_given_x(x)
        log_prob = self.reconstruct_log_prob(q_z_given_x, x)
        kl_div = self.kl_div(q_z_given_x, self.p_z())
        return log_prob - kl_div, log_prob.detach(), kl_div.detach(), q_z_given_x

    def sample(self, batch_size):
        z = self.p_z().sample(batch_size)
        decode = self.p_x_given_z(z)
        return decode.sample()

    def forward(self, z):
        return self.p_x_given_z(z)

    def reconstruct_log_prob(self, q_z_given_x, x):
        z_logits = q_z_given_x.base_dist.logits
        z = nn.functional.gumbel_softmax(z_logits, hard=True)
        nz = z.flatten(-2)
        reconstruct_log_prob = self.p_x_given_z(nz).log_prob(x)
        return reconstruct_log_prob

def tb_vae_reconstruct(tb_writer, images):
    def _fn(trainer_state):
        z = trainer_state.trainable.q_z_given_x(images).sample()
        reconstruct_images = trainer_state.trainable.p_x_given_z(z.flatten(-2)).sample()
        imglist = torch.cat([images, reconstruct_images], dim=0)
        grid_image = make_grid(imglist, padding=10, nrow=10)
        tb_writer.add_image("reconstruct_image", grid_image, trainer_state.epoch_num)
    return _fn


parser = argparse.ArgumentParser(description='PyGen MNIST Discrete VAE')
parser.add_argument("--datasets_folder", default="~/datasets")
parser.add_argument("--tb_folder", default=None)
parser.add_argument("--device", default="cpu")
parser.add_argument("--max_epoch", default=10, type=int)
parser.add_argument("--num_states", default=10, type=int)
parser.add_argument("--num_vars", default=1, type=int)
parser.add_argument("--dummy_run", action="store_true")
parser.add_argument("--decoder_type", default="simple_pixelcnn")
ns = parser.parse_args()

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(), lambda x: (x > 0.5).float(),
    train.DevicePlacement()])
mnist_dataset = torchvision.datasets.MNIST(
    ns.datasets_folder, train=True, download=False,
    transform=transform)
train_dataset, validation_dataset = random_split(mnist_dataset, [50000, 10000])
torch.set_default_device(ns.device)
match ns.decoder_type:
    case "simple_pixelcnn":
        intermediate_channels = 8
        net = pixelcnn_layer.make_simple_pixelcnn_net()
        conditional_sp_distribution = pixelcnn_layer.make_pixelcnn_layer(
            pixelcnn_dist.make_bernoulli_base_distribution(), net, [1, 28, 28], intermediate_channels)
        decoder_type = nn.Sequential(
            pixelcnn_layer.SpatialExpand(ns.num_vars*ns.num_states, intermediate_channels, [28, 28]),
            conditional_sp_distribution)
    case "basic":
        decoder_type = nn.Sequential(
            nn.Linear(ns.num_vars * ns.num_states, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 784),
            bernoulli_layer.IndependentBernoulli([1, 28, 28]))
    case _:
        raise RuntimeError(f"decoder_type {ns.decoder_type} not recognised.")

digit_distribution = VAE(ns.num_vars, ns.num_states, decoder_type)
example_valid_images = next(iter(torch.utils.data.DataLoader(validation_dataset, batch_size=10)))[0]
tb_writer = SummaryWriter(ns.tb_folder)
epoch_end_callbacks_list = [
    callbacks.tb_epoch_log_metrics(tb_writer),
    callbacks.tb_dataset_metrics_logging(tb_writer, "validation", validation_dataset),
    tb_vae_reconstruct(tb_writer, example_valid_images)
    ]
epoch_end_callbacks = callbacks.callback_compose(epoch_end_callbacks_list)
train.train(digit_distribution, train_dataset, pygen_models_train.vae_objective(),
    batch_end_callback=callbacks.tb_batch_log_metrics(tb_writer),
    epoch_end_callback=epoch_end_callbacks, dummy_run=ns.dummy_run, max_epoch=ns.max_epoch, epoch_regularizer=False)
