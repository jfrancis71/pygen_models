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
        self.q_z_given_x = nn.Sequential(
            classifier_net.ClassifierNet(mnist=True, num_classes=self.num_states),
            layer_one_hot_categorical.OneHotCategorical())
        intermediate_channels = 3
        net = pixelcnn_layer.make_simple_pixelcnn_net()
        conditional_sp_distribution = pixelcnn_layer.make_pixelcnn_layer(
            pixelcnn_dist.make_bernoulli_base_distribution(), net, [1, 28, 28], intermediate_channels)
        self.p_x_given_z = nn.Sequential(
            pixelcnn_layer.SpatialExpand(num_states, intermediate_channels, [28, 28]),
            conditional_sp_distribution)
        self.logits_p_z = torch.zeros([self.num_states])
        self.identity_matrix = torch.eye(num_states).float()

    def p_z(self):
        return OneHotCategorical(logits=self.logits_p_z)

    def kl_div(self, p, q):
        kl_div = torch.sum(p.probs * (p.logits - q.logits), axis=-1)
        return kl_div

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

    def p_z_given_x(self, x):  # Warning, this is expensive. linear in self.num_states. Feasible if num_states small.
        batch_size = x.shape[0]
        reconstruct_log_prob_z = torch.stack([
            self.p_x_given_z(self.identity_matrix[z].unsqueeze(0).repeat(batch_size, 1)).log_prob(x)
            for z in range(self.num_states)], dim=1)
        joint_z = self.p_z().logits + reconstruct_log_prob_z
        return OneHotCategorical(logits=joint_z)

    def kl_gap(self, x):  # Warning, this is expensive. linear in self.num_states. Feasible if num_states small.
        q_z_given_x = self.q_z_given_x(x)
        p_z_given_x = self.p_z_given_x(x)
        kl_div = self.kl_div(q_z_given_x, p_z_given_x)
        return kl_div

    def forward(self, z):
        return self.p_x_given_z(z)


class VAEAnalytic(VAE):
    def __init__(self, num_states):
        super().__init__(num_states)

    def reconstruct_log_prob(self, q_z_given_x, x):
        batch_size = x.shape[0]
        p_x_given_z = self.p_x_given_z(self.identity_matrix.unsqueeze(0).repeat(batch_size, 1, 1))
        logits_p_x_given_z = p_x_given_z.log_prob(x.unsqueeze(1).repeat(1, self.num_states, 1, 1, 1))
        log_prob = torch.sum(logits_p_x_given_z * q_z_given_x.probs, axis=1)
        return log_prob


class VAEUniform(VAE):
    def __init__(self, num_states, num_z_samples):
        super().__init__(num_states)
        self.num_z_samples = num_z_samples

    def reconstruct_log_prob(self, q_z_given_x, x):
        batch_size = x.shape[0]
        z = OneHotCategorical(logits=torch.zeros([batch_size, self.num_states])).sample([self.num_z_samples])
        reconstruct_log_prob = self.p_x_given_z(z).log_prob(x.unsqueeze(0).repeat(self.num_z_samples, 1, 1, 1, 1))
        importance_sample = self.num_states * q_z_given_x.log_prob(z).exp() * reconstruct_log_prob
        log_prob = importance_sample.mean(axis=0)
        return log_prob


class VAEReinforce(VAE):
    def __init__(self, num_states, num_z_samples):
        super().__init__(num_states)
        self.num_z_samples = num_z_samples

    def reconstruct_log_prob(self, q_z_given_x, x):
        z = q_z_given_x.sample([self.num_z_samples])
        reconstruct_log_prob = self.p_x_given_z(z).log_prob(x.unsqueeze(0).repeat(self.num_z_samples, 1, 1, 1, 1))
        reinforce = q_z_given_x.log_prob(z) * reconstruct_log_prob.detach()
        log_prob = torch.mean(reconstruct_log_prob + reinforce - reinforce.detach(), axis=0)
        return log_prob


class VAEReinforceBaseline(VAE):
    def __init__(self, num_states, num_z_samples):
        super().__init__(num_states)
        net = pixelcnn_dist.make_simple_pixelcnn_net()
        self.baseline_dist = pixelcnn_dist.make_pixelcnn(pixelcnn_dist.make_bernoulli_base_distribution(), net,
            event_shape=[1, 28, 28])
        self.num_z_samples = num_z_samples

    def reconstruct_log_prob(self, q_z_given_x, x):
        baseline_log_prob = self.baseline_dist.log_prob(x)
        z = q_z_given_x.sample([self.num_z_samples])
        reconstruct_log_prob = self.p_x_given_z(z).log_prob(x.unsqueeze(0).repeat(self.num_z_samples, 1, 1, 1, 1))
        reinforce = q_z_given_x.log_prob(z) * (reconstruct_log_prob-baseline_log_prob).detach()
        log_prob = torch.mean(reconstruct_log_prob + reinforce - reinforce.detach(), axis=0)
        return log_prob + baseline_log_prob - baseline_log_prob.detach()


class VAEReinforceGumbel(VAE):
    def __init__(self, num_states, num_z_samples):
        super().__init__(num_states)
        self.num_z_samples = num_z_samples
        self.gumbel_dist = torch.distributions.gumbel.Gumbel(torch.tensor(0.0), torch.tensor(1.0))

    def reconstruct_log_prob(self, q_z_given_x, x):
        batch_size = x.shape[0]
        z = q_z_given_x.sample([self.num_z_samples])
        reconstruct_log_prob = self.p_x_given_z(z).log_prob(x.unsqueeze(0).repeat(self.num_z_samples, 1, 1, 1, 1))
        gumbels = self.gumbel_dist.sample([self.num_z_samples, batch_size, self.num_states])
        encode_h = torch.softmax(q_z_given_x.probs.unsqueeze(0).repeat(self.num_z_samples, 1, 1) + gumbels, dim=-1)
        with torch.no_grad():
            decode_h = self.p_x_given_z(encode_h)
        log_prob_h = decode_h.log_prob(x.unsqueeze(0).repeat(self.num_z_samples, 1, 1, 1, 1))
        reinforce = q_z_given_x.log_prob(z) * (reconstruct_log_prob.detach()-log_prob_h)
        reparam = log_prob_h
        log_prob = (reconstruct_log_prob + reinforce - reinforce.detach() + reparam - reparam.detach()).mean(axis=0)
        return log_prob


def tb_vae_reconstruct(tb_writer, images):
    def _fn(trainer_state):
        z = trainer_state.trainable.q_z_given_x(images).sample()
        reconstruct_images = trainer_state.trainable.p_x_given_z(z).sample()
        imglist = torch.cat([images, reconstruct_images], dim=0)
        grid_image = make_grid(imglist, padding=10, nrow=10)
        tb_writer.add_image("reconstruct_image", grid_image, trainer_state.epoch_num)
    return _fn


def tb_vae_posteriors(tb_writer, images):
    def _fn(trainer_state):
        probs_q_z_given_x = trainer_state.trainable.q_z_given_x(images).probs.detach()
        tb_writer.add_image("example_qz_x", probs_q_z_given_x.unsqueeze(0).cpu().numpy(), trainer_state.epoch_num)
        probs_p_z_given_x = trainer_state.trainable.p_z_given_x(images).probs.detach()
        tb_writer.add_image("example_pz_x", probs_p_z_given_x.unsqueeze(0).cpu().numpy(), trainer_state.epoch_num)
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

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(), lambda x: (x > 0.5).float(),
    train.DevicePlacement()])
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
    epoch_end_callback=epoch_end_callbacks, dummy_run=ns.dummy_run, max_epoch=ns.max_epoch, epoch_regularizer=False)
