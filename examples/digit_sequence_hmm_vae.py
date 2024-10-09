import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.utils import make_grid
import pygen.train.train as train
from pygen.neural_nets import classifier_net
import pygen.train.callbacks as callbacks
import pygen.layers.independent_bernoulli as bernoulli_layer
from pygen_models.datasets import sequential_mnist
import pixelcnn_pp.model as pixelcnn_model
import pygen_models.layers.pixelcnn as pixelcnn
import pyro.nn
import pygen_models.distributions.made as made
import pygen_models.layers.made as lmade


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        num_z = 128
        net = pixelcnn_model.PixelCNN(nr_resnet=1, nr_filters = 160, input_channels = 1,
            nr_params = 1, nr_conditional = 160)
        channel_layer = bernoulli_layer.IndependentBernoulli(event_shape=[1])
        self.pixelcnn_layer = nn.Sequential(
            pixelcnn.SpatialExpand(num_z, 160, [28, 28]),
            pixelcnn.PixelCNN(net, [1, 28, 28], channel_layer, 160))
        self.encoder = classifier_net.ClassifierNet(mnist=True, num_classes=num_z)
        net = pyro.nn.AutoRegressiveNN(num_z, [num_z * 2, num_z * 2], param_dims=[1],
                                       permutation=torch.arange(num_z))
        self.made = made.MadeBernoulli(net, num_z, None)
        lnet = pyro.nn.ConditionalAutoRegressiveNN(num_z, num_z, [num_z*2, num_z*2], param_dims=[1], permutation=torch.arange(num_z))
        self.lmade = lmade.Made(lnet, num_z)
        self.fwd = nn.Sequential(nn.ReLU(), nn.Linear(128, 128))
        self.bwd = nn.Sequential(nn.ReLU(), nn.Linear(128, 128))

    def train(self, x):
        z_logits1 = self.bwd(self.encoder(x[:, 1]))
        one_hot_logits1 = torch.stack([z_logits1*0.0, z_logits1], dim=-1)
        one_hot1 = nn.functional.gumbel_softmax(one_hot_logits1, hard=True)
        z1 = one_hot1[..., 1]
        log_prob1 = self.pixelcnn_layer(z1).log_prob(x[:, 0])

        z_logits2 = self.fwd(self.encoder(x[:, 0]))
        one_hot_logits2 = torch.stack([z_logits2*0.0, z_logits2], dim=-1)
        one_hot2 = nn.functional.gumbel_softmax(one_hot_logits2, hard=True)
        z2 = one_hot2[..., 1]
        log_prob2 = self.pixelcnn_layer(z2).log_prob(x[:, 1])

        z_logits3 = self.fwd(self.encoder(x[:, 1]))
        one_hot_logits3 = torch.stack([z_logits3*0.0, z_logits3], dim=-1)
        one_hot3 = nn.functional.gumbel_softmax(one_hot_logits3, hard=True)
        z3 = one_hot3[..., 1]
        log_prob3 = self.pixelcnn_layer(z3).log_prob(x[:, 2])

        log_prob = log_prob1 + log_prob2 + log_prob3

        made_log = self.pz().log_prob(z1.detach())
        lmade_log1 = self.lmade(z1.detach()).log_prob(z2.detach())
        lmade_log2 = self.lmade(z2.detach()).log_prob(z3.detach())
        lmade_log = lmade_log1 + lmade_log2

#        kl_div1 = torch.distributions.bernoulli.Bernoulli(logits=z_logits1).log_prob(z1).sum(axis=1)*0.0 - made_log
#        kl_div2 = torch.distributions.bernoulli.Bernoulli(logits=z_logits2).log_prob(z2).sum(axis=1)*0.0 - lmade_log
#        kl_div = kl_div1 + kl_div2

#        return (log_prob - kl_div + kl_div.detach(),
#            made_log.detach(), lmade_log.detach())
        return (log_prob + made_log - made_log.detach() + lmade_log - lmade_log.detach(),
                made_log.detach(), lmade_log.detach())

    def pz(self):
        return self.made


def reconstruct_cb(model, image_seq):
    def _fn():
        z_logits = model.fwd(model.encoder(image_seq[:, 0]))
        one_hot_logits = torch.stack([z_logits*0.0, z_logits], dim=-1)
        one_hot = nn.functional.gumbel_softmax(one_hot_logits, hard=True)
        z = one_hot[..., 1]
        dist = model.pixelcnn_layer(z)
        sample = dist.sample()
        imglist = torch.cat([image_seq[:, 0], sample], dim=0)
        grid_image = make_grid(imglist, padding=10, nrow=10)
        return grid_image
    return _fn


def gen_made_cb(model):
    def _fn():
        z = model.made.sample(sample_shape=[10]).float()
        dist = model.pixelcnn_layer(z)
        samples = dist.sample()
        z1 = model.lmade(z).sample(sample_shape=[]).float()
        dist1 = model.pixelcnn_layer(z1)
        samples1 = dist1.sample()

        z2 = model.lmade(z1).sample(sample_shape=[]).float()
        dist2 = model.pixelcnn_layer(z2)
        samples2 = dist2.sample()

        z3 = model.lmade(z2).sample(sample_shape=[]).float()
        dist3 = model.pixelcnn_layer(z3)
        samples3 = dist3.sample()

        z4 = model.lmade(z3).sample(sample_shape=[]).float()
        dist4 = model.pixelcnn_layer(z4)
        samples4 = dist4.sample()

        pic = torch.cat([samples, samples1, samples2, samples3, samples4])
        grid_image = make_grid(pic, padding=10, nrow=10)
        return grid_image
    return _fn


def train_objective(distribution, batch):
    log_prob, made, lmade = distribution.train(batch[0])
    log_prob_mean = log_prob.mean()
    made_mean = made.mean()
    lmade_mean = lmade.mean()
    metrics_data = (log_prob_mean.cpu().detach().numpy(),
                    made_mean.cpu().detach().numpy(), lmade_mean.cpu().detach().numpy())
    metrics_dtype = [('log_prob', 'float32'), ('made', 'float32'), ('lmade', 'float32')]
    metrics = np.array(metrics_data, metrics_dtype)
    return log_prob_mean, metrics


parser = argparse.ArgumentParser(description='PyGen MNIST Sequence HMM')
parser.add_argument("--datasets_folder", default="~/datasets")
parser.add_argument("--tb_folder", default=None)
parser.add_argument("--device", default="cpu")
parser.add_argument("--max_epoch", default=10, type=int)
parser.add_argument("--dummy_run", action="store_true")
ns = parser.parse_args()

torch.set_default_device(ns.device)

seq_len = 3
batch_size = 16
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), lambda x: (x > 0.5).float(),
    train.DevicePlacement()])
mnist_dataset = torchvision.datasets.MNIST(
    ns.datasets_folder, train=True, download=False,
    transform=transform)
train_mnist_dataset, validation_mnist_dataset = random_split(mnist_dataset, [55000, 5000],
    generator=torch.Generator(device=torch.get_default_device()))
train_dataset = sequential_mnist.SequentialMNISTDataset(train_mnist_dataset, seq_len, ns.dummy_run)
validation_dataset = sequential_mnist.SequentialMNISTDataset(validation_mnist_dataset, seq_len, ns.dummy_run)

model = Model()
example_valid_images = next(iter(torch.utils.data.DataLoader(validation_dataset, batch_size=10)))
tb_writer = SummaryWriter(ns.tb_folder)
epoch_end_callbacks = callbacks.callback_compose([
    callbacks.tb_log_image(tb_writer, "reconstructed_image_seq", reconstruct_cb(model, example_valid_images[0])),
    callbacks.tb_log_image(tb_writer, "gen_made_seq", gen_made_cb(model)),
    callbacks.tb_epoch_log_metrics(tb_writer),
    callbacks.tb_dataset_metrics_logging(tb_writer, "validation", validation_dataset, batch_size=batch_size)
])
if __name__ == "__main__":
    train.train(model, train_dataset, train_objective,
        batch_end_callback=callbacks.tb_batch_log_metrics(tb_writer), model_path="./model.pth",
        epoch_end_callback=epoch_end_callbacks, dummy_run=ns.dummy_run, max_epoch=ns.max_epoch, epoch_regularizer=False, batch_size=batch_size)
