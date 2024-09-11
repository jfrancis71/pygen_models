"""Simple Discrete VAE for training on MNIST."""


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
import pygen.layers.independent_categorical as independent_categorical
import pygen_models.distributions.discrete_vae as discrete_vae
import pygen_models.train.train as pygen_models_train
import pygen_models.train.callbacks as pygen_models_callbacks
import pygen_models.layers.pixelcnn as pixelcnn
from pygen_models.neural_nets import simple_pixelcnn_net


class IndependentLatentModel(nn.Module):
    def __init__(self, num_vars, num_states, decoder_type):
        super().__init__()
        self.num_vars = num_vars
        self.num_states = num_states
        self.p_z_logits = nn.Parameter(torch.zeros([self.num_vars, self.num_states], requires_grad=True))
        match decoder_type:
            case "simple_pixelcnn":
                num_pixelcnn_params = 8
                channel_layer = bernoulli_layer.IndependentBernoulli(event_shape=[1])
                net = simple_pixelcnn_net.SimplePixelCNNNet(1, channel_layer.params_size(), num_pixelcnn_params)
                self.p_x_given_z = nn.Sequential(nn.Flatten(),
                    pixelcnn.SpatialExpand(ns.num_vars * ns.num_states, num_pixelcnn_params, [28, 28]),
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
        base_dist = torch.distributions.categorical.Categorical(logits=self.p_z_logits)
        p_z_dist = torch.distributions.independent.Independent(base_distribution=base_dist, reinterpreted_batch_ndims=1)
        return p_z_dist

def tb_vae_reconstruct(vae, images):
    def _fn():
        z = vae.q_z_given_x(images).sample()
        one_hot_z = nn.functional.one_hot(z, vae.latent_model.num_states).float()
        flatten_z = one_hot_z.flatten(-2)
        reconstruct_images = vae.latent_model.p_x_given_z(flatten_z).sample()
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
parser.add_argument("--num_states", default=8, type=int)
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
    classifier_net.ClassifierNet(mnist=True, num_classes=ns.num_states*ns.num_vars),
    independent_categorical.IndependentCategorical([ns.num_vars], ns.num_states))
latent_model = IndependentLatentModel(ns.num_vars, ns.num_states, ns.decoder_type)
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
