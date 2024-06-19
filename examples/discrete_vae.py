"""Simple Discrete VAE with just one categorical variable over 10 states for training on MNIST.
Provides analytic, uniform and reinforce methods for computing gradient on reconstruction.
"""


import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
import torchvision
from pygen.train import train
from pygen.train import callbacks
from pygen.neural_nets import classifier_net
import pygen.layers.categorical as layer_categorical
import torch.nn as nn
import pygen.layers.independent_bernoulli as bernoulli_layer
from torch.distributions.categorical import Categorical


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 784)
        self.distribution_layer = bernoulli_layer.IndependentBernoulli(event_shape=[1, 28, 28])

    def forward(self, x):
        x = self.linear(x)
        return self.distribution_layer(x)


class VAE(nn.Module):
    def __init__(self, mode, num_z_samples):
        super().__init__()
        self.encoder = classifier_net.ClassifierNet(mnist=True)
        self.decoder = Decoder()
        self.layer = bernoulli_layer.IndependentBernoulli(event_shape=[1, 28, 28])
        self.num_z_samples = num_z_samples
        self.mode = mode

    def kl_div(self, log_encoder):
        p = Categorical(logits=torch.zeros([10], device=next(self.parameters()).device))
        q = Categorical(logits=log_encoder)
        kl_div = torch.sum(q.probs * (q.logits - p.logits), axis=1)
        return kl_div

    def reg(self, log_encoder):
        q = Categorical(logits=log_encoder).logits
        reg = -torch.sum(q, axis=1)
        return reg

    def sample_reconstruct_log_prob_analytic(self, log_encoder, x):
        device = next(self.parameters()).device
        batch_size = x.shape[0]
        log_prob1 = torch.stack([self.decoder(
            nn.functional.one_hot(
                torch.tensor(z).to(device), 10).unsqueeze(0).repeat(batch_size, 1).to(device).float()).log_prob(x)
            for z in range(10)], dim=1)
        soft = torch.softmax(log_encoder, dim=1)
        arr = soft * log_prob1
        log_prob = torch.sum(arr, axis=1)
        return log_prob

    def sample_reconstruct_log_prob_reinforce(self, log_encoder, x):  # Classic REINFORCE
        z = torch.distributions.categorical.Categorical(logits=log_encoder).sample()
        encode = torch.nn.functional.one_hot(z, 10).float()
        decode = self.decoder(encode)
        log_prob = decode.log_prob(x)
        reinforce = log_encoder[torch.arange(log_encoder.size(0)), z] * log_prob.detach()
        return log_prob + reinforce - reinforce.detach()

    def sample_reconstruct_log_prob_uniform(self, log_encoder, x):  # Using uniform sampling
        device = next(self.parameters()).device
        batch_size = x.shape[0]
        z = torch.distributions.categorical.Categorical(logits=torch.zeros([batch_size, 10]).to(device)).sample()
        probs = torch.softmax(log_encoder, dim=1)
        encode = torch.nn.functional.one_hot(z, 10).float()
        decode = self.decoder(encode)
        log_prob = decode.log_prob(x)
        reinforce = 10.0 * probs[torch.arange(log_encoder.size(0)), z] * log_prob
        return reinforce

    def sample_reconstruct_log_prob(self, log_encoder, x):
        if self.mode == "analytic":
            return self.sample_reconstruct_log_prob_analytic(log_encoder, x)
        elif self.mode == "uniform":
            return self.sample_reconstruct_log_prob_uniform(log_encoder, x)
        elif self.mode == "reinforce":
            return self.sample_reconstruct_log_prob_reinforce(log_encoder, x)
        else:
            raise RuntimeError("Unknown reconstruct mode: ", self.mode)

    def log_prob(self, x):
        return self.elbo(x)[0]

    def elbo(self, x):
        log_encoder = self.encoder(x)
        log_probs = [self.sample_reconstruct_log_prob(log_encoder, x) for _ in range(self.num_z_samples)]
        log_prob = torch.mean(torch.stack(log_probs), dim=0)
        kl_div = self.kl_div(log_encoder)
        reg = self.reg(log_encoder)
        self.loss = reg
        return log_prob - kl_div, log_prob.detach(), kl_div.detach()

    def sample(self, batch_size):
        device = next(self.parameters()).device
        z = torch.distributions.categorical.Categorical(probs=torch.ones([10]) / 10).sample(batch_size).to(device)
        encode = torch.nn.functional.one_hot(z, 10).float()
        decode = self.decoder(encode)
        return decode.sample()

    def forward(self, z):
        encode = torch.nn.functional.one_hot(z, 10).float().unsqueeze(0)
        decode = self.decoder.linear(encode)
        return self.decoder.distribution_layer(decode[0])


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
parser.add_argument("--sample_mode", default="analytic")
parser.add_argument("--num_z_samples", default=10, type=int)
parser.add_argument("--dummy_run", action="store_true")
ns = parser.parse_args()

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), lambda x: (x > 0.5).float()])
mnist_dataset = torchvision.datasets.MNIST(
    ns.datasets_folder, train=True, download=False,
    transform=transform)
train_dataset, validation_dataset = random_split(mnist_dataset, [50000, 10000])
class_labels = [f"{num}" for num in range(10)]

digit_distribution = VAE(ns.sample_mode, ns.num_z_samples).to(ns.device)
tb_writer = SummaryWriter(ns.tb_folder)

epoch_end_callbacks = callbacks.callback_compose([
    callbacks.TBConditionalImagesCallback(tb_writer, "z_conditioned_images"),
    callbacks.TBTotalLogProbCallback(tb_writer, "train_epoch_log_prob"),
    callbacks.TBDatasetLogProbDistributionCallback(tb_writer, "validation_log_prob", validation_dataset),
    TBDatasetVAECallback(tb_writer, "validation", validation_dataset)
])

VAETrainer(
    digit_distribution.to(ns.device),
    train_dataset, max_epoch=ns.max_epoch,
    batch_end_callback=callbacks.TBBatchLogProbCallback(tb_writer, "batch_log_prob"),
    epoch_end_callback=epoch_end_callbacks, dummy_run=ns.dummy_run).train()
