"""Simple Discrete VAE with just one categorical variable for training on MNIST.
Provides analytic, uniform and reinforce methods for computing gradient on reconstruction.

Results default params, validation epoch: (Analytic -93.2, Uniform -111.3, ReinforceBaseline -94.25, Gumbel -95.77)
"""


import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from torch.distributions.one_hot_categorical import OneHotCategorical
import torchvision
from torchvision.utils import make_grid
from pygen.train import callbacks
from pygen.neural_nets import classifier_net
import pygen.train.train as train
import pygen_models.layers.pixelcnn as pixelcnn_layer
import pygen_models.layers.one_hot_categorical as layer_one_hot_categorical
import pygen_models.distributions.pixelcnn as pixelcnn_dist
import pygen_models.train.train as pygen_models_train


class VAE(nn.Module):
    def __init__(self, num_states):
        super().__init__()
        self.num_states = num_states
        self.encoder = nn.Sequential(classifier_net.ClassifierNet(mnist=True, num_classes=self.num_states), layer_one_hot_categorical.OneHotCategorical())
        intermediate_channels = 3
        net = pixelcnn_layer.make_simple_pixelcnn_net()
        conditional_sp_distribution = pixelcnn_layer.make_pixelcnn_layer(pixelcnn_dist.make_bernoulli_base_distribution(), net, [1, 28, 28], intermediate_channels)
        self.decoder = nn.Sequential(pixelcnn_layer.SpatialExpand(num_states, intermediate_channels, [28, 28]), conditional_sp_distribution)
        self.pz_logits = torch.zeros([self.num_states])
        self.identity_matrix = torch.eye(num_states).float()

    def pz(self):
        return OneHotCategorical(logits=self.pz_logits)

    def kl_div(self, encoder_dist):
        kl_div = torch.sum(encoder_dist.probs * (encoder_dist.logits - self.pz().logits), axis=1)
        return kl_div

    def log_prob(self, x):
        return self.elbo(x)[0]

    def elbo(self, x):
        encoder_dist = self.encoder(x)
        log_prob = self.reconstruct_log_prob(encoder_dist, x)
        kl_div = self.kl_div(encoder_dist)
        return log_prob - kl_div, log_prob.detach(), kl_div.detach()

    def sample(self, batch_size):
        z = self.pz().sample(batch_size)
        decode = self.decoder(z)
        return decode.sample()

    def pz_given_x(self, x):  # Warning, this is expensive. linear in self.num_states. Feasible if num_states small.
        batch_size = x.shape[0]
        reconstruct_log_prob_z = torch.stack([
            self.decoder(self.identity_matrix[z].unsqueeze(0).repeat(batch_size, 1)).log_prob(x)
            for z in range(self.num_states)], dim=1)
        joint_z = self.pz().logits + reconstruct_log_prob_z
        pz_given_x = torch.distributions.categorical.Categorical(logits=joint_z).logits
        return pz_given_x

    def kl_gap(self, x):  # Warning, this is expensive. linear in self.num_states. Feasible if num_states small.
        encoder_dist = self.encoder(x)
        pz_given_x = self.pz_given_x(x)
        kl_div = torch.sum(encoder_dist.probs * (encoder_dist.logits - pz_given_x), axis=1)
        return kl_div

    def forward(self, z):
        return self.decoder(z)


class VAEAnalytic(VAE):
    def __init__(self, num_states):
        super().__init__(num_states)

    def reconstruct_log_prob(self, encoder_dist, x):
        batch_size = x.shape[0]
        reconstruct_log_prob_z = torch.stack([
            self.decoder(self.identity_matrix[z].unsqueeze(0).repeat(batch_size, 1)).log_prob(x)
            for z in range(self.num_states)], dim=1)
        recon = encoder_dist.probs * reconstruct_log_prob_z
        log_prob = torch.sum(recon, axis=1)
        return log_prob


class VAEMultiSample(VAE):
    def __init__(self, num_states, num_z_samples):
        super().__init__(num_states)
        self.num_z_samples = num_z_samples

    def reconstruct_log_prob(self, q_dist, x):
        log_probs = [self.sample_reconstruct_log_prob(q_dist, x) for _ in range(self.num_z_samples)]
        log_prob = torch.mean(torch.stack(log_probs), dim=0)
        return log_prob


class VAEUniform(VAEMultiSample):
    def __init__(self, num_states, num_z_samples):
        super().__init__(num_states, num_z_samples)

    def sample_reconstruct_log_prob(self, encoder_dist, x):
        batch_size = x.shape[0]
        z = torch.distributions.one_hot_categorical.OneHotCategorical(logits=torch.zeros([batch_size, self.num_states])).sample()
        reconstruct_log_prob = self.decoder(z).log_prob(x)
        importance_sample = self.num_states * encoder_dist.log_prob(z).exp() * reconstruct_log_prob
        return importance_sample


class VAEReinforce(VAEMultiSample):
    def __init__(self, num_states, num_z_samples):
        super().__init__(num_states, num_z_samples)

    def sample_reconstruct_log_prob(self, encoder_dist, x):
        z = encoder_dist.sample()
        reconstruct_log_prob = self.decoder(z).log_prob(x)
        reinforce = encoder_dist.log_prob(z) * reconstruct_log_prob.detach()
        return reconstruct_log_prob + reinforce - reinforce.detach()


class VAEReinforceBaseline(VAE):
    def __init__(self, num_states, num_z_samples):
        super().__init__(num_states)
        net = pixelcnn_dist.make_simple_pixelcnn_net()
        self.baseline_dist = pixelcnn_dist.make_pixelcnn(pixelcnn_dist.make_bernoulli_base_distribution(), net,
            event_shape=[1, 28, 28])
        self.num_z_samples = num_z_samples

    def reconstruct_log_prob(self, q_dist, x):
        baseline_log_prob = self.baseline_dist.log_prob(x)
        log_probs = [self.sample_reconstruct_log_prob(q_dist, x, baseline_log_prob) for _ in range(self.num_z_samples)]
        log_prob = torch.mean(torch.stack(log_probs), dim=0)
        return log_prob + baseline_log_prob - baseline_log_prob.detach()

    def sample_reconstruct_log_prob(self, encoder_dist, x, baseline_log_prob):
        z = encoder_dist.sample()
        reconstruct_log_prob = self.decoder(z).log_prob(x)
        reinforce = encoder_dist.log_prob(z) * (reconstruct_log_prob.detach()-baseline_log_prob.detach())
        return reconstruct_log_prob + reinforce - reinforce.detach()


class VAEReinforceGumbel(VAEMultiSample):
    def __init__(self, num_states, num_z_samples):
        super().__init__(num_states, num_z_samples)
        self.gumbel_dist = torch.distributions.gumbel.Gumbel(torch.tensor(0.0), torch.tensor(1.0))

    def sample_reconstruct_log_prob(self, encoder_dist, x):
        z = encoder_dist.sample()
        reconstruct_log_prob = self.decoder(z).log_prob(x)
        gumbels = self.gumbel_dist.sample(encoder_dist.probs.shape)
        encode_h = torch.softmax(encoder_dist.probs + gumbels, dim=-1)
        with torch.no_grad():
            decode_h = self.decoder(encode_h)
        log_prob_h = decode_h.log_prob(x)
        reinforce = encoder_dist.log_prob(z) * (reconstruct_log_prob.detach()-log_prob_h)
        reparam = log_prob_h
        return reconstruct_log_prob + reinforce - reinforce.detach() + reparam - reparam.detach()


def tb_vae_reconstruct(tb_writer, images):
    def _fn(trainer_state):
        z = trainer_state.trainable.encoder(images).sample()
        reconstruct_images = trainer_state.trainable.decoder(z).sample()
        imglist = torch.cat([images, reconstruct_images], dim=0)
        grid_image = make_grid(imglist, padding=10, nrow=10)
        tb_writer.add_image("reconstruct_image", grid_image, trainer_state.epoch_num)
    return _fn

def tb_vae_posteriors(tb_writer, images):
    def _fn(trainer_state):
        qz_x = trainer_state.trainable.encoder(images).probs
        tb_writer.add_image("example_qz_x", qz_x.detach().unsqueeze(0).cpu().numpy(), trainer_state.epoch_num)
        pz_x = trainer_state.trainable.pz_given_x(images).exp()
        tb_writer.add_image("example_pz_x", pz_x.detach().unsqueeze(0).cpu().numpy(), trainer_state.epoch_num)
    return _fn


parser = argparse.ArgumentParser(description='PyGen MNIST Discrete VAE')
parser.add_argument("--datasets_folder", default="~/datasets")
parser.add_argument("--tb_folder", default=None)
parser.add_argument("--device", default="cpu")
parser.add_argument("--max_epoch", default=10, type=int)
parser.add_argument("--mode", default="analytic")
parser.add_argument("--num_states", default=10, type=int)
parser.add_argument("--num_z_samples", default=10, type=int)
parser.add_argument("--dummy_run", action="store_true")
parser.add_argument("--log_kl_gap", action="store_true")
parser.add_argument("--log_posteriors", action="store_true")
ns = parser.parse_args()

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), lambda x: (x > 0.5).float(), train.DevicePlacement()])
mnist_dataset = torchvision.datasets.MNIST(
    ns.datasets_folder, train=True, download=False,
    transform=transform)
train_dataset, validation_dataset = random_split(mnist_dataset, [50000, 10000])
torch.set_default_device(ns.device)
match ns.mode:
    case "analytic":
        digit_distribution = VAEAnalytic(ns.num_states)
    case "uniform":
        digit_distribution = VAEUniform(ns.num_states, ns.num_z_samples)
    case "reinforce":
        digit_distribution = VAEReinforce(ns.num_states, ns.num_z_samples)
    case "reinforce_baseline":
        digit_distribution = VAEReinforceBaseline(ns.num_states, ns.num_z_samples)
    case "reinforce_gumbel":
        digit_distribution = VAEReinforceGumbel(ns.num_states, ns.num_z_samples)
    case _:
        raise RuntimeError(f"mode {ns.mode} not recognised.")

example_valid_images = next(iter(torch.utils.data.DataLoader(validation_dataset, batch_size=10)))[0]
tb_writer = SummaryWriter(ns.tb_folder)
epoch_end_callbacks_list = [
    callbacks.tb_conditional_images(tb_writer, "z_conditioned_images", num_labels=ns.num_states),
    callbacks.tb_epoch_log_metrics(tb_writer),
    callbacks.tb_dataset_metrics_logging(tb_writer, "validation", validation_dataset),
    tb_vae_reconstruct(tb_writer, example_valid_images)
    ]
if ns.log_posteriors:
    epoch_end_callbacks_list.append(tb_vae_posteriors(tb_writer, example_valid_images))
epoch_end_callbacks = callbacks.callback_compose(epoch_end_callbacks_list)
train.train(digit_distribution, train_dataset, pygen_models_train.vae_objective(ns.log_kl_gap),
    batch_end_callback=callbacks.tb_batch_log_metrics(tb_writer),
    epoch_end_callback=epoch_end_callbacks, dummy_run=ns.dummy_run, epoch_regularizer=False)
