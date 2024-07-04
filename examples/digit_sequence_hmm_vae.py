import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical
import torchvision
import pygen.train.callbacks as callbacks
from pygen.train import train
import pygen_models.distributions.hmm as hmm
import pygen.layers.independent_bernoulli as bernoulli_layer
import pygen_models.distributions.pixelcnn as pixelcnn_dist
import torch.nn.functional as F
from pygen.neural_nets import classifier_net
from pygen_models.datasets import sequential_mnist


class MaskedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.mask = nn.Parameter(torch.ones([out_channels, in_channels, 3, 3]).float(), requires_grad=False)
        self.mask[:, :, 2, 2] = 0.0
        self.mask[:, :, 1, 1:] = 0.0
        self.mask[:, :, 2] = 0.0

    def forward(self, x):
        weight = self.conv.weight * self.mask.detach()
        return nn.functional.conv2d(x, weight, bias=self.conv.bias, stride=self.stride, padding=self.padding)


class SimplePixelCNNNetwork(nn.Module):
    def __init__(self, num_conditional):
        super().__init__()
        self.conv1 = MaskedConv2d(1, 32, 1, 1)
        if num_conditional is not None:
            self.prj1 = nn.Linear(num_conditional, 32*28*28)
        self.conv2 = MaskedConv2d(32, 1, 1, 1)

    def forward(self, x, sample=False, conditional=None):
        x = self.conv1(x)
        if conditional is not None:
            prj = self.prj1(conditional[:,:,0,0]).reshape([-1, 32, 28, 28])
            x = x + prj
        x = F.relu(x)
        x = self.conv2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, num_states):
        super().__init__()
        in_channels = 1
        mid_channels = 9216
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(mid_channels, 128)
        self.fc2 = nn.Linear(128, num_states)

    # pylint: disable=C0103,C0116
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)  # pylint: disable=E1101
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        distribution = torch.distributions.categorical.Categorical(logits=x)
        return distribution


class HMMVAE(nn.Module):
    def __init__(self, mode, num_states, num_z_samples, device):
        super().__init__()
        self.mode = mode
        self.num_states = num_states
        self.num_z_samples = num_z_samples
        self.device = device
        self.pixelcnn_net = SimplePixelCNNNetwork(self.num_states)
        self.baseline_pixelcnn_net = SimplePixelCNNNetwork(self.num_states)
        self.base_layer = bernoulli_layer.IndependentBernoulli(event_shape=[1])
        self.baseline_dist = pixelcnn_dist._PixelCNN(
            self.baseline_pixelcnn_net,
            [1, 28, 28],
            self.base_layer, None
            )
        self.q = Encoder(num_states)
        self.prior_states_vector = nn.Parameter(torch.randn([num_states]))  # (state)
        self.state_transitions_matrix = \
            nn.Parameter(torch.randn(num_states, num_states))  # (state, state')

    def kl_div_cat(self, p, q):
        kl_div = torch.sum(p.probs * (p.logits - q.logits), axis=1)
        return kl_div

    def sample_reconstruct_log_prob_analytic(self, q_dist, x):
        """encoder_dist is tensor of shape [B, T, N], x has shape [B, T, C, Y, X]"""
        log_prob = torch.zeros([x.shape[0]], device=self.device)
        for t in range(x.shape[1]):
            for s in range(self.num_states):
                log_prob += torch.exp(q_dist[:, t, s]) * self.state_emission_distribution(s).log_prob(x[:, t])
        return log_prob

    def sample_reconstruct_log_prob_uniform(self, q_dist, x):  # Using uniform sampling
        batch_size = q_dist.shape[0]
        log_prob = torch.zeros([x.shape[0]], device=self.device)
        for t in range(x.shape[1]):
            z = torch.distributions.categorical.Categorical(logits=torch.zeros([batch_size, self.num_states]).to(self.device)).sample()
            encode = torch.nn.functional.one_hot(z, self.num_states).float()
            q_probs = torch.exp(q_dist[:,t])[torch.arange(batch_size), z]
            log_prob_t = self.state_emission_distribution(z).log_prob(x[:, t])
            log_prob += self.num_states * q_probs * log_prob_t
        return log_prob


    def sample_reconstruct_log_prob_reinforce(self, q_dist, x):  # Using reinforce sampling
        batch_size = q_dist.shape[0]
        log_prob = torch.zeros([x.shape[0]], device=self.device)
        for t in range(x.shape[1]):
            z = torch.distributions.categorical.Categorical(logits=q_dist[:,t]).sample()
            encode = torch.nn.functional.one_hot(z, self.num_states).float()
            q_logits = q_dist[:,t][torch.arange(batch_size), z]
            log_prob_t = self.state_emission_distribution(z).log_prob(x[:, t])
            reinforce = (log_prob_t-self.log_prob_baseline[:, t]).detach() * q_logits
            log_prob += log_prob_t + reinforce - reinforce.detach()
        return log_prob

    def sample_reconstruct_log_prob(self, encoder_dist, x):
        if self.mode == "analytic":
            return self.sample_reconstruct_log_prob_analytic(encoder_dist, x)
        elif self.mode == "uniform":
            return self.sample_reconstruct_log_prob_uniform(encoder_dist, x)
        elif self.mode == "reinforce":
            return self.sample_reconstruct_log_prob_reinforce(encoder_dist, x)
        else:
            raise RuntimeError("Unknown reconstruct mode: ", self.mode)

    def kl_div(self, q_dist):
        kl_div = self.kl_div_cat(Categorical(logits=q_dist[:, 0]), self.prior_state_distribution())
        for t in range(1, q_dist.shape[1]):
            for s in range(self.num_states):
                kl_div += torch.exp(q_dist[:, t - 1, s]) * self.kl_div_cat(Categorical(logits=q_dist[:, t]),
                                                                           Categorical(logits=
                                                                                       self.state_transition_distribution().logits[
                                                                                           s]))
        return kl_div

    def elbo(self, x):
        q_dist = torch.zeros([x.shape[0], x.shape[1], self.num_states], device=self.device)
        self.log_prob_baseline = torch.zeros([x.shape[0], x.shape[1]], device=self.device)
        for t in range(x.shape[1]):
            q_dist[:,t] = self.q(x[:, t]).logits
            self.log_prob_baseline[:, t] = self.baseline_dist.log_prob(x[:, t])
        log_probs = [self.sample_reconstruct_log_prob(q_dist, x) for _ in range(self.num_z_samples)]
        log_prob = torch.mean(torch.stack(log_probs), dim=0)
        kl_div = self.kl_div(q_dist)
        log_prob_baseline = torch.sum(self.log_prob_baseline, axis=1)
        return log_prob - kl_div + log_prob_baseline - log_prob_baseline.detach()

    def log_prob(self, x):
        return self.elbo(x)

    def state_emission_distribution(self, s):
        encode = torch.nn.functional.one_hot(torch.tensor(s).to(self.device), self.num_states).float()
        if len(encode.shape) == 1:
            encode = encode.unsqueeze(-1).unsqueeze(-1).repeat(1, 28, 28)
        else:
            encode = encode.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 28, 28)
        dist = pixelcnn_dist._PixelCNN(
            self.pixelcnn_net,
            [1, 28, 28],
            self.base_layer,
            encode
            )
        return dist

    def sample(self, num_steps):
        """Samples observation sequence."""
        return self.sample_variables(num_steps)[1]

    def sample_variables(self, num_steps):
        """Samples state sequence and observation sequence and returns as a tuple."""
        state = self.prior_state_distribution().sample()
        state_sequence = [state]
        observation = self.state_emission_distribution(state).sample()
        observation_sequence = [observation]
        for _ in range(num_steps-1):
            state = Categorical(logits=self.state_transitions_matrix[state]).sample()
            observation = self.state_emission_distribution(state).sample()
            state_sequence.append(state)
            observation_sequence.append(observation)
        state_sequence = torch.stack(state_sequence, dim=0)
        observation_sequence = torch.stack(observation_sequence, dim=0)
        # pylint: disable=E1101
        return state_sequence, observation_sequence

    def forward(self, z):
        return self.state_emission_distribution(z)

    def prior_state_distribution(self):
        return Categorical(logits=self.prior_states_vector)

    def state_transition_distribution(self):
        return Categorical(logits=self.state_transitions_matrix)


class HMMTrainer(train.DistributionTrainer):
    # pylint: disable=R0913
    def __init__(self, trainable, dataset, batch_size=32, max_epoch=10, batch_end_callback=None,
                 epoch_end_callback=None, use_scheduler=False, dummy_run=False, model_path=None):
        super().__init__(
            trainable, dataset, batch_size, max_epoch, batch_end_callback,
            epoch_end_callback, use_scheduler=use_scheduler, dummy_run=dummy_run,
            model_path=model_path)

    def kl_div(self, observation):  # x is B*Event_shape, ie just one element of sequence
        pz_given_x = self.trainable.q(observation)
        pz = Categorical(logits=pz_given_x.logits*0.0)
        kl_div = self.trainable.kl_div_cat(pz, pz_given_x)
        return kl_div

    def batch_log_prob(self, batch):
        log_prob = self.trainable.log_prob(batch[0].to(self.device)) - \
            self.kl_div(batch[0][:, 0].to(self.device))/(self.epoch+1)
        return log_prob


class TBSequenceImageCallback:
    # pylint: disable=R0903
    def __init__(self, tb_writer, tb_name):
        self.tb_writer = tb_writer
        self.tb_name = tb_name

    def __call__(self, trainer):
        sample_size = 8
        num_steps = 3
        imglist = [trainer.trainable.sample(num_steps=num_steps) for _ in range(sample_size)]
        imglist = torch.clip(torch.cat(imglist, axis=0), 0.0, 1.0)  # pylint: disable=E1101
        grid_image = torchvision.utils.make_grid(imglist, padding=10, nrow=num_steps)
        self.tb_writer.add_image(self.tb_name, grid_image, trainer.epoch)


class TBSequenceTransitionMatrixCallback:
    # pylint: disable=R0903
    def __init__(self, tb_writer, tb_name):
        self.tb_writer = tb_writer
        self.tb_name = tb_name

    def __call__(self, trainer):
        image = trainer.trainable.state_transition_distribution().probs.detach().unsqueeze(0).cpu().numpy()
        self.tb_writer.add_image(self.tb_name, image, trainer.epoch)


parser = argparse.ArgumentParser(description='PyGen MNIST Sequence HMM')
parser.add_argument("--datasets_folder", default=".")
parser.add_argument("--tb_folder", default=None)
parser.add_argument("--device", default="cpu")
parser.add_argument("--max_epoch", default=10, type=int)
parser.add_argument("--mode", default="cpu")
parser.add_argument("--num_states", default=10, type=int)
parser.add_argument("--num_z_samples", default=10, type=int)
parser.add_argument("--dummy_run", action="store_true")
ns = parser.parse_args()

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), lambda x: (x > 0.5).float()])
mnist_dataset = torchvision.datasets.MNIST(
    ns.datasets_folder, train=True, download=False,
    transform=transform)
train_mnist_dataset, validation_mnist_dataset = random_split(mnist_dataset, [55000, 5000])
train_dataset = sequential_mnist.SequentialMNISTDataset(train_mnist_dataset, ns.dummy_run)
validation_dataset = sequential_mnist.SequentialMNISTDataset(validation_mnist_dataset, ns.dummy_run)

mnist_hmm = HMMVAE(ns.mode, ns.num_states, ns.num_z_samples, ns.device)

tb_writer = SummaryWriter(ns.tb_folder)
epoch_end_callbacks = callbacks.callback_compose([
    callbacks.TBConditionalImagesCallback(tb_writer, "z_conditioned_images", num_labels=ns.num_states),
    callbacks.TBDatasetLogProbDistributionCallback(tb_writer, "validation_log_prob", validation_dataset),
    TBSequenceImageCallback(tb_writer, tb_name="image_sequence"),
    TBSequenceTransitionMatrixCallback(tb_writer, tb_name="state_transition"),
])
trainer = HMMTrainer(
    mnist_hmm.to(ns.device),
    train_dataset, max_epoch=ns.max_epoch, epoch_end_callback=epoch_end_callbacks, dummy_run=ns.dummy_run)
torch.autograd.set_detect_anomaly(True)
trainer.train()
