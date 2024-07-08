"""Simple Discrete VAE with just one categorical variable over 10 states for training on MNIST.
Provides analytic, uniform and reinforce methods for computing gradient on reconstruction.
"""


import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from torch.distributions.categorical import Categorical
import torchvision
from pygen.train import train
from pygen.train import callbacks
from pygen.neural_nets import classifier_net
import pygen.layers.categorical as layer_categorical
import pygen.layers.independent_bernoulli as bernoulli_layer
import pygen_models.layers.pixelcnn as pixelcnn_layer
from pygen_models.neural_nets import simple_pixel_cnn_net


class LayerPixelCNN(pixelcnn_layer._PixelCNNDistribution):
    def __init__(self, num_conditional):
        pixelcnn_net = simple_pixel_cnn_net.SimplePixelCNNNetwork(num_conditional)
        base_layer = bernoulli_layer.IndependentBernoulli(event_shape=[1])
        super().__init__(pixelcnn_net, [1, 28, 28], base_layer, num_conditional)

    def forward(self, x):
        return super().forward(x.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 28, 28))


class VAE(nn.Module):
    def __init__(self, num_states):
        super().__init__()
        self.num_states = num_states
        self.encoder = classifier_net.ClassifierNet(mnist=True, num_classes=self.num_states)
        self.decoder = LayerPixelCNN(self.num_states)
        self.layer = bernoulli_layer.IndependentBernoulli(event_shape=[1, 28, 28])
        self.pz_logits = torch.zeros([self.num_states])

    def pz(self):
        return Categorical(logits=self.pz_logits.to(self.device()))

    def kl_div(self, encoder_dist):
        kl_div = torch.sum(encoder_dist.probs * (encoder_dist.logits - self.pz().logits), axis=1)
        return kl_div

    def reg(self, encoder_dist):
        reg = -torch.sum(encoder_dist.logits, axis=1)
        return reg

    def log_prob(self, x):
        return self.elbo(x)[0]

    def elbo(self, x):
        encoder_dist = self.encoder(x)
        log_prob = self.reconstruct_log_prob(encoder_dist, x)
        kl_div = self.kl_div(encoder_dist)
        reg = self.reg(encoder_dist)
        self.loss = reg
        return log_prob - kl_div, log_prob.detach(), kl_div.detach()

    def sample(self, batch_size):
        z = self.pz().sample(batch_size)
        encode = torch.nn.functional.one_hot(z, self.num_states).float()
        decode = self.decoder(encode)
        return decode.sample()

    def forward(self, z):
        encode = torch.nn.functional.one_hot(z, self.num_states).float().unsqueeze(0)
        return self.decoder(encode)

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


class VAETrainer(train.DistributionTrainer):
    def __init__(self, trainable, dataset, batch_size=32, max_epoch=10, batch_end_callback=None,
                 epoch_end_callback=None, use_scheduler=False, dummy_run=False, model_path=None):
        super().__init__(
            trainable, dataset, batch_size, max_epoch, batch_end_callback,
            epoch_end_callback, use_scheduler=use_scheduler, dummy_run=dummy_run,
            model_path=model_path)

    def batch_log_prob(self, batch):
        log_prob = self.trainable.log_prob(batch[0].to(self.device)) - \
            self.trainable.loss/(self.epoch+1)
        return log_prob


class TBDatasetVAECallback:
    def __init__(self, tb_writer, tb_name, dataset, batch_size=32):
        self.tb_writer = tb_writer
        self.tb_name = tb_name
        self.batch_size = batch_size
        self.dataset = dataset

    def __call__(self, trainer):
        dataloader = torch.utils.data.DataLoader(
            self.dataset, collate_fn=None,
            batch_size=self.batch_size, shuffle=True, drop_last=True)
        log_reconstruct = 0.0
        kl = 0.0
        size = 0
        for (_, batch) in enumerate(dataloader):
            _, batch_log_reconstruct, batch_kl = trainer.trainable.elbo(batch[0].to(trainer.device))
            log_reconstruct += batch_log_reconstruct.mean().item()
            kl += batch_kl.mean().item()
            size += 1
        self.tb_writer.add_scalar(self.tb_name+"_reconstruct", log_reconstruct/size, trainer.epoch)
        self.tb_writer.add_scalar(self.tb_name + "_kl", kl / size, trainer.epoch)


parser = argparse.ArgumentParser(description='PyGen MNIST Discrete VAE')
parser.add_argument("--datasets_folder", default=".")
parser.add_argument("--tb_folder", default=None)
parser.add_argument("--device", default="cpu")
parser.add_argument("--max_epoch", default=10, type=int)
parser.add_argument("--mode", default="analytic")
parser.add_argument("--num_states", default=10, type=int)
parser.add_argument("--num_z_samples", default=10, type=int)
parser.add_argument("--dummy_run", action="store_true")
ns = parser.parse_args()

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), lambda x: (x > 0.5).float()])
mnist_dataset = torchvision.datasets.MNIST(
    ns.datasets_folder, train=True, download=False,
    transform=transform)
train_dataset, validation_dataset = random_split(mnist_dataset, [50000, 10000])

match ns.mode:
    case "analytic":
        digit_distribution = VAEAnalytic(ns.num_states)
    case "uniform":
        digit_distribution = VAEUniform(ns.num_states, ns.num_z_samples)
    case "reinforce":
        digit_distribution = VAEReinforce(ns.num_states, ns.num_z_samples)
    case _:
        raise RuntimeError("mode {} not recognised.".format(ns.mode))

tb_writer = SummaryWriter(ns.tb_folder)
epoch_end_callbacks = callbacks.callback_compose([
    callbacks.TBConditionalImagesCallback(tb_writer, "z_conditioned_images", num_labels=ns.num_states),
    callbacks.TBTotalLogProbCallback(tb_writer, "train_epoch_log_prob"),
    callbacks.TBDatasetLogProbDistributionCallback(tb_writer, "validation_log_prob", validation_dataset),
    TBDatasetVAECallback(tb_writer, "validation", validation_dataset)
])

VAETrainer(
    digit_distribution.to(ns.device),
    train_dataset, max_epoch=ns.max_epoch,
    batch_end_callback=callbacks.TBBatchLogProbCallback(tb_writer, "batch_log_prob"),
    epoch_end_callback=epoch_end_callbacks, dummy_run=ns.dummy_run).train()
