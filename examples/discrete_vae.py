"""Simple Discrete VAE with just one categorical variable for training on MNIST.
Provides analytic, uniform and reinforce methods for computing gradient on reconstruction.
"""


import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from torch.distributions.categorical import Categorical
import torchvision
from torchvision.utils import make_grid
from pygen.train import callbacks
from pygen.neural_nets import classifier_net
import pygen.layers.categorical as layer_categorical
import pygen.layers.independent_bernoulli as bernoulli_layer
import pygen.train.train as train
import pygen_models.layers.pixelcnn as pixelcnn_layer
from pygen_models.neural_nets import simple_pixel_cnn_net
import pygen_models.distributions.pixelcnn as pixelcnn_dist
import pygen_models.train.callbacks as pygen_models_callbacks
import pygen_models.train.train as pygen_models_train


class LayerPixelCNN(pixelcnn_layer._PixelCNNDistribution):
    def __init__(self, num_conditional):
        pixelcnn_net = simple_pixel_cnn_net.SimplePixelCNNNetwork(num_conditional)
        base_layer = bernoulli_layer.IndependentBernoulli(event_shape=[1])
        super().__init__(pixelcnn_net, [1, 28, 28], base_layer, num_conditional)


class VAE(nn.Module):
    def __init__(self, num_states):
        super().__init__()
        self.num_states = num_states
        self.encoder = nn.Sequential(classifier_net.ClassifierNet(mnist=True, num_classes=self.num_states), layer_categorical.Categorical())
        self.decoder = LayerPixelCNN(self.num_states)
        self.pz_logits = torch.zeros([self.num_states])

    def pz(self):
        return Categorical(logits=self.pz_logits.to(self.device()))

    def kl_div(self, encoder_dist):
        kl_div = torch.sum(encoder_dist.probs * (encoder_dist.logits - self.pz().logits), axis=1)
        return kl_div

    def epoch_regularizer_penalty(self, batch):
        encoder_dist = self.encoder(batch[0])
        reg = -torch.sum(encoder_dist.logits, axis=1)
        return reg.mean()

    def log_prob(self, x):
        return self.elbo(x)[0]

    def elbo(self, x):
        encoder_dist = self.encoder(x)
        log_prob = self.reconstruct_log_prob(encoder_dist, x)
        kl_div = self.kl_div(encoder_dist)
        return log_prob - kl_div, log_prob.detach(), kl_div.detach()

    def sample(self, batch_size):
        z = self.pz().sample(batch_size)
        encode = torch.nn.functional.one_hot(z, self.num_states).float()
        decode = self.decoder(encode)
        return decode.sample()

    def forward(self, z):
        return self.decoder(z)

    def device(self):
        return next(self.parameters()).device


class VAEAnalytic(VAE):
    def __init__(self, num_states):
        super().__init__(num_states)

    def reconstruct_log_prob(self, encoder_dist, x):
        batch_size = x.shape[0]
        log_prob1 = torch.stack([self.decoder(
            nn.functional.one_hot(
                torch.tensor(z, device=self.device()), self.num_states).unsqueeze(0).repeat(batch_size, 1).float()).log_prob(x)
            for z in range(self.num_states)], dim=1)
        arr = encoder_dist.probs * log_prob1
        log_prob = torch.sum(arr, axis=1)
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
        z = torch.distributions.categorical.Categorical(logits=torch.zeros([batch_size, self.num_states], device=self.device())).sample()
        encode = torch.nn.functional.one_hot(z, self.num_states).float()
        decode = self.decoder(encode)
        log_prob = decode.log_prob(x)
        importance_sample = self.num_states * encoder_dist.probs[torch.arange(encoder_dist.probs.size(0)), z] * log_prob
        return importance_sample


class VAEReinforce(VAEMultiSample):
    def __init__(self, num_states, num_z_samples):
        super().__init__(num_states, num_z_samples)

    def sample_reconstruct_log_prob(self, encoder_dist, x):
        z = encoder_dist.sample()
        encode = torch.nn.functional.one_hot(z, self.num_states).float()
        decode = self.decoder(encode)
        log_prob = decode.log_prob(x)
        log_encoder = encoder_dist.logits
        reinforce = log_encoder[torch.arange(encoder_dist.probs.size(0)), z] * log_prob.detach()
        return log_prob + reinforce - reinforce.detach()


class VAEReinforceBaseline(VAEMultiSample):
    def __init__(self, num_states, num_z_samples):
        super().__init__(num_states, num_z_samples)
        self.baseline_pixelcnn_net = simple_pixel_cnn_net.SimplePixelCNNNetwork(self.num_states)
        self.base_layer = bernoulli_layer.IndependentBernoulli(event_shape=[1])
        self.baseline_dist = pixelcnn_dist._PixelCNN(
            self.baseline_pixelcnn_net,
            [1, 28, 28],
            self.base_layer, None
        )

    def reconstruct_log_prob(self, q_dist, x):
        baseline_log_prob = self.baseline_dist.log_prob(x)
        log_probs = [self.sample_reconstruct_log_prob(q_dist, x, baseline_log_prob) for _ in range(self.num_z_samples)]
        log_prob = torch.mean(torch.stack(log_probs), dim=0)
        return log_prob + baseline_log_prob - baseline_log_prob.detach()

    def sample_reconstruct_log_prob(self, encoder_dist, x, baseline_log_prob):
        z = encoder_dist.sample()
        encode = torch.nn.functional.one_hot(z, self.num_states).float()
        decode = self.decoder(encode)
        log_prob = decode.log_prob(x)
        log_encoder = encoder_dist.logits
        reinforce = log_encoder[torch.arange(encoder_dist.probs.size(0)), z] * (log_prob.detach()-baseline_log_prob.detach())
        return log_prob + reinforce - reinforce.detach()


class VAEReinforceGumbel(VAEMultiSample):
    def __init__(self, num_states, num_z_samples):
        super().__init__(num_states, num_z_samples)
        self.gumbel = torch.distributions.gumbel.Gumbel(torch.tensor(0.0), torch.tensor(1.0))

    def sample_reconstruct_log_prob(self, encoder_dist, x):
        z = encoder_dist.sample()
        encode_H = torch.nn.functional.one_hot(z, self.num_states).float()
        decode_H = self.decoder(encode_H)
        log_prob_H = decode_H.log_prob(x)
        gumbels = torch.distributions.gumbel.Gumbel(torch.tensor(0.0, device=self.device()), torch.tensor(1.0, device=self.device())).sample(encoder_dist.probs.shape)
        encode_h = torch.softmax(encoder_dist.probs + gumbels, dim=-1)
        with torch.no_grad():
            decode_h = self.decoder(encode_h)
        log_prob_h = decode_h.log_prob(x)
        log_encoder = encoder_dist.logits
        reinforce = log_encoder[torch.arange(encoder_dist.probs.size(0)), z] * (log_prob_H.detach()-log_prob_h)
        reparam = log_prob_h
        return log_prob_H + reinforce - reinforce.detach() + reparam - reparam.detach()


class TBVAEReconstructCallback:
    def __init__(self, tb_writer, dataset):
        self.tb_writer = tb_writer
        self.dataset = dataset

    def __call__(self, training_loop_info):
        dataset_images = torch.stack([self.dataset[i][0] for i in range(10)])
        z = training_loop_info.trainable.encoder(dataset_images).sample()
        reconstruct_images = training_loop_info.trainable.decoder(nn.functional.one_hot(z, training_loop_info.trainable.num_states).float()).sample()
        imglist = torch.cat([dataset_images, reconstruct_images], dim=0)
        grid_image = make_grid(imglist, padding=10, nrow=10)
        self.tb_writer.add_image("reconstruct_image", grid_image, training_loop_info.epoch_num)


parser = argparse.ArgumentParser(description='PyGen MNIST Discrete VAE')
parser.add_argument("--datasets_folder", default="~/datasets")
parser.add_argument("--tb_folder", default=None)
parser.add_argument("--device", default="cpu")
parser.add_argument("--max_epoch", default=10, type=int)
parser.add_argument("--mode", default="analytic")
parser.add_argument("--num_states", default=10, type=int)
parser.add_argument("--num_z_samples", default=10, type=int)
parser.add_argument("--dummy_run", action="store_true")
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

tb_writer = SummaryWriter(ns.tb_folder)
epoch_end_callbacks = callbacks.callback_compose([
    callbacks.TBConditionalImages(tb_writer, "z_conditioned_images", num_labels=ns.num_states),
    callbacks.TBEpochLogMetrics(tb_writer),
    callbacks.TBDatasetMetricsLogging(tb_writer, "validation", validation_dataset),
    TBVAEReconstructCallback(tb_writer, validation_dataset)
])
train.train(digit_distribution, train_dataset, pygen_models_train.vae_trainer,
    batch_end_callback=callbacks.TBBatchLogMetrics(tb_writer),
    epoch_end_callback=epoch_end_callbacks, dummy_run=ns.dummy_run, epoch_regularizer=True)
