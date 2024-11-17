"""Simple Discrete VAE for training on MNIST."""


import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
import torch.distributions.kl as kl_div_mod
import torchvision
from torchvision.utils import make_grid
from pygen.train import callbacks
from pygen.neural_nets import classifier_net
import pygen.train.train as train
import pygen.layers.independent_bernoulli as bernoulli_layer
import pygen_models.distributions.discrete_vae as discrete_vae
import pygen_models.distributions.r_independent_bernoulli as r_ind_bern
import pygen_models.train.train as pygen_models_train
import pygen_models.train.callbacks as pygen_models_callbacks
import pygen_models.layers.pixelcnn as pixelcnn
from pygen_models.neural_nets import simple_pixelcnn_net
from pygen_models.distributions.made import MadeBernoulli
import pyro.nn


class RIndependentBernoulliLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return r_ind_bern.RIndependentBernoulliDistribution(logits=x)


class IndependentLatentModel(nn.Module):
    def __init__(self, num_vars, decoder_type):
        super().__init__()
        self.num_vars = num_vars
        self.net = pyro.nn.AutoRegressiveNN(num_vars, [num_vars*2, num_vars*2], param_dims=[1], permutation=torch.arange(num_vars))
        match decoder_type:
            case "simple_pixelcnn":
                num_pixelcnn_params = 8
                channel_layer = bernoulli_layer.IndependentBernoulli(event_shape=[1])
                net = simple_pixelcnn_net.SimplePixelCNNNet(1, channel_layer.params_size(), num_pixelcnn_params)
                self.p_x_given_z = nn.Sequential(
                    pixelcnn.SpatialExpand(ns.num_vars, num_pixelcnn_params, [28, 28]),
                    pixelcnn.PixelCNN(net, [1, 28, 28], channel_layer, num_pixelcnn_params))
            case "basic":
                self.p_x_given_z = nn.Sequential(nn.Flatten(),
                    nn.Linear(ns.num_vars * ns.num_states, 256), nn.ReLU(),
                    nn.Linear(256, 512), nn.ReLU(),
                    nn.Linear(512, 784),
                    bernoulli_layer.IndependentBernoulli([1, 28, 28]))
            case _:
                raise RuntimeError(f"decoder_type {ns.decoder_type} not recognised.")

    def p_z(self):
        p_z_dist = MadeBernoulli(self.net, self.num_vars)
        return p_z_dist


@kl_div_mod.register_kl(r_ind_bern.RIndependentBernoulliDistribution, MadeBernoulli)
def kl_div_r_independent_bernoulli_made_bernoulli(p, q):
    sample_z = p.rsample()
    kl_div = p.log_prob(sample_z) - q.log_prob(sample_z)
    return kl_div


def tb_vae_reconstruct(vae, images):
    def _fn():
        z = vae.q_z_given_x(images).sample()
        reconstruct_images = vae.latent_model.p_x_given_z(z).sample()
        imglist = torch.cat([images, reconstruct_images], dim=0)
        grid_image = make_grid(imglist, padding=10, nrow=10)
        return grid_image
    return _fn


parser = argparse.ArgumentParser(description='PyGen MNIST Discrete VAE')
parser.add_argument("--datasets_folder", default="~/datasets")
parser.add_argument("--tb_folder", default=None)
parser.add_argument("--images_folder", default=None)
parser.add_argument("--device", default="cpu")
parser.add_argument("--max_epoch", default=10, type=int)
parser.add_argument("--num_vars", default=20, type=int)
parser.add_argument("--beta", default=1.0, type=float)
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

encoder = nn.Sequential(
    classifier_net.ClassifierNet(mnist=True, num_classes=ns.num_vars),
    RIndependentBernoulliLayer()
    )
latent_model = IndependentLatentModel(ns.num_vars, ns.decoder_type)
digit_distribution = discrete_vae.DiscreteVAE(latent_model, encoder, ns.beta)

example_valid_images = next(iter(torch.utils.data.DataLoader(validation_dataset, batch_size=10)))[0]
tb_writer = SummaryWriter(ns.tb_folder)
epoch_end_callbacks_list = [
    callbacks.tb_epoch_log_metrics(tb_writer),
    callbacks.tb_dataset_metrics_logging(tb_writer, "validation", validation_dataset),
    callbacks.tb_log_image(tb_writer, "reconstruct_images", tb_vae_reconstruct(digit_distribution, example_valid_images)),
    callbacks.tb_log_image(tb_writer, "generated_images", pygen_models_callbacks.sample_images(digit_distribution))
]
if ns.images_folder is not None:
    epoch_end_callbacks_list.append(callbacks.file_log_image(ns.images_folder, "reconstruct_images", tb_vae_reconstruct(digit_distribution, example_valid_images)))
    epoch_end_callbacks_list.append(callbacks.file_log_image(ns.images_folder, "generated_images", pygen_models_callbacks.sample_images(digit_distribution)))
epoch_end_callbacks = callbacks.callback_compose(epoch_end_callbacks_list)
train.train(digit_distribution, train_dataset, pygen_models_train.vae_objective(),
    batch_end_callback=callbacks.tb_batch_log_metrics(tb_writer),
    epoch_end_callback=epoch_end_callbacks, dummy_run=ns.dummy_run, max_epoch=ns.max_epoch, epoch_regularizer=False)
