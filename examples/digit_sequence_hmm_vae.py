import argparse
import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import torch.distributions.kl as kl_div_mod
import torchvision
from torchvision.utils import make_grid
import pyro.nn
import pygen.train.train as train
from pygen.neural_nets import classifier_net
import pygen.train.callbacks as callbacks
import pygen.layers.independent_bernoulli as bernoulli_layer
from pygen_models.datasets import sequential_mnist
import pixelcnn_pp.model as pixelcnn_model
import pygen_models.layers.pixelcnn as pixelcnn
import pygen_models.distributions.made as made
import pygen_models.distributions.r_independent_bernoulli as r_ind_bern
import pygen_models.layers.made as lmade
import pygen_models.distributions.discrete_vae as discrete_vae
import pygen_models.train.train as pygen_models_train


class MadeHMM(nn.Module):
    def __init__(self):
        super().__init__()
        num_z = 128
        made_net = pyro.nn.AutoRegressiveNN(num_z, [num_z * 2, num_z * 2], param_dims=[1],
                                       permutation=torch.arange(num_z))
        self.made = made.MadeBernoulli(made_net, num_z, None)
        made_layer_net = pyro.nn.ConditionalAutoRegressiveNN(num_z, num_z, [num_z * 2, num_z * 2], param_dims=[1],
                                                   permutation=torch.arange(num_z))
        self.lmade = lmade.Made(made_layer_net, num_z)

    def log_prob(self, z):
        made_log = self.made.log_prob(z[:,0].detach())
        lmade_log1 = self.lmade(z[:,0].detach()).log_prob(z[:, 1].detach())
        lmade_log2 = self.lmade(z[:, 1].detach()).log_prob(z[:, 2].detach())
        log = made_log + lmade_log1 + lmade_log2
        return log

    def sample(self, sample_shape=[]):
        z1 = self.made.sample(sample_shape)
        z2 = self.lmade(z1).sample([])
        z3 = self.lmade(z2).sample([])
        return torch.stack([z1, z2, z3], dim=1)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        num_z = 128
        self.encoder = classifier_net.ClassifierNet(mnist=True, num_classes=num_z)
        self.fwd = nn.Sequential(nn.ReLU(), nn.Linear(128, 128))
        self.bwd = nn.Sequential(nn.ReLU(), nn.Linear(128, 128))

    def forward(self, x):
        z_logits1 = self.bwd(self.encoder(x[:, 1]))
        z_logits2 = self.fwd(self.encoder(x[:, 0]))
        z_logits3 = self.fwd(self.encoder(x[:, 1]))
        base_dist = r_ind_bern.RIndependentBernoulliDistribution(logits=torch.stack([z_logits1, z_logits2, z_logits3], dim=1))
        dist = torch.distributions.independent.Independent(base_distribution=base_dist, reinterpreted_batch_ndims=1)
        return dist


class SeqObservationModel(nn.Module):
    def __init__(self, seq_layer, z):
        super().__init__()
        self.seq_layer = seq_layer
        self.z = z

    def log_prob(self, x):
        l1 = self.seq_layer(self.z[:, 0]).log_prob(x[:, 0])
        l2 = self.seq_layer(self.z[:, 1]).log_prob(x[:, 1])
        l3 = self.seq_layer(self.z[:, 2]).log_prob(x[:, 2])
        return l1 + l2 + l3

    def sample(self):
        s1 = self.seq_layer(self.z[:, 0]).sample()
        s2 = self.seq_layer(self.z[:, 1]).sample()
        s3 = self.seq_layer(self.z[:, 2]).sample()
        sample = torch.stack([s1, s2, s3], dim=1)
        return sample


class IndependentLatentModel(nn.Module):
    def __init__(self):
        super().__init__()
        num_z = 128
        self.made_hmm = MadeHMM()
        observation_net = pixelcnn_model.PixelCNN(nr_resnet=1, nr_filters = 160, input_channels = 1,
            nr_params = 1, nr_conditional = 160)
        channel_layer = bernoulli_layer.IndependentBernoulli(event_shape=[1])
        self.p_x_given_z_seq = nn.Sequential(
            pixelcnn.SpatialExpand(num_z, 160, [28, 28]),
            pixelcnn.PixelCNN(observation_net, [1, 28, 28], channel_layer, 160))

    def p_x_given_z(self, z):
        return SeqObservationModel(self.p_x_given_z_seq, z)

    def p_z(self):
        return self.made_hmm


@kl_div_mod.register_kl(torch.distributions.independent.Independent, MadeHMM)
def kl_div_r_independent_bernoulli_made_hmm(p, q):
    sample_z = p.sample()
    kl_div = p.log_prob(sample_z).detach() - q.log_prob(sample_z)
    return kl_div


def gen_made_cb(model):
    def _fn():
        z = model.latent_model.p_z().sample(sample_shape=[10]).float()
        sample = model.latent_model.p_x_given_z(z).sample()
        grid_image = make_grid(sample.flatten(start_dim=0, end_dim=1), padding=10, nrow=3)
        return grid_image
    return _fn


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

latent_model = IndependentLatentModel()
encoder = Encoder()
model = discrete_vae.DiscreteVAE(latent_model, encoder, 1.0)
example_valid_images = next(iter(torch.utils.data.DataLoader(validation_dataset, batch_size=10)))
tb_writer = SummaryWriter(ns.tb_folder)
epoch_end_callbacks = callbacks.callback_compose([
#    callbacks.tb_log_image(tb_writer, "reconstructed_image_seq", reconstruct_cb(model, example_valid_images[0])),
    callbacks.tb_log_image(tb_writer, "gen_made_seq", gen_made_cb(model)),
    callbacks.tb_epoch_log_metrics(tb_writer),
    callbacks.tb_dataset_metrics_logging(tb_writer, "validation", validation_dataset, batch_size=batch_size)
])
if __name__ == "__main__":
    train.train(model, train_dataset, pygen_models_train.vae_objective(),
        batch_end_callback=callbacks.tb_batch_log_metrics(tb_writer), model_path="./model.pth",
        epoch_end_callback=epoch_end_callbacks, dummy_run=ns.dummy_run, max_epoch=ns.max_epoch, epoch_regularizer=False, batch_size=batch_size)
