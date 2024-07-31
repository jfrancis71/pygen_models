import argparse
import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical
import torchvision
import pygen.train.train as train
import pygen.train.callbacks as callbacks
from pygen.neural_nets import classifier_net
import pygen_models.layers.pixelcnn as pixelcnn_layer
import pygen_models.layers.one_hot_categorical as layer_one_hot_categorical
import pygen_models.distributions.pixelcnn as pixelcnn_dist
import pygen_models.distributions.hmm as pygen_hmm
from pygen_models.datasets import sequential_mnist
import pygen_models.train.train as pygen_models_train
import pygen_models.train.callbacks as pygen_models_callbacks


class HMMVAE(pygen_hmm.HMM):
    def __init__(self, num_steps, num_states):
        conditional_sp_distribution = pixelcnn_layer.make_pixelcnn_layer(
            pixelcnn_dist.make_bernoulli_base_distribution(),
            pixelcnn_layer.make_simple_pixelcnn_net(), [1, 28, 28], ns.num_states)
        layer_pixelcnn_bernoulli = nn.Sequential(pixelcnn_layer.SpatialExpand(ns.num_states, ns.num_states,
                                                                              [28, 28]), conditional_sp_distribution)
        super().__init__(num_steps, num_states, layer_pixelcnn_bernoulli)
        self.q = nn.Sequential(classifier_net.ClassifierNet(mnist=True, num_classes=self.num_states),
            layer_one_hot_categorical.OneHotCategorical())

    def log_prob(self, x):
        return self.elbo(x)

    def elbo(self, x):
        q_dist = torch.zeros([x.shape[0], x.shape[1], self.num_states], device=self.device())
        for t in range(x.shape[1]):
            q_dist[:, t] = self.q(x[:, t]).logits
        reconstruct_log_prob = self.reconstruct_log_prob(q_dist, x)
        kl_div = self.kl_div(q_dist)
        return reconstruct_log_prob - kl_div, reconstruct_log_prob, kl_div, Categorical(q_dist[0])

    def kl_div(self, q_dist):
        kl_div = self.kl_div_cat(Categorical(logits=q_dist[:, 0]), self.prior_state_distribution())
        for t in range(1, q_dist.shape[1]):
            for s in range(self.num_states):
                kl_div += torch.exp(q_dist[:, t - 1, s]) * self.kl_div_cat(Categorical(logits=q_dist[:, t]),
                    Categorical(logits=self.state_transition_distribution().logits[s]))
        return kl_div

    def kl_div_cat(self, p, q):
        kl_div = torch.sum(p.probs * (p.logits - q.logits), axis=1)
        return kl_div

    def reg(self, observations):  # x is B*Event_shape, ie just one element of sequence
        pz_given_x = self.q(observations[:, 0])
        pz = Categorical(logits=pz_given_x.logits*0.0)
        kl_div = self.kl_div_cat(pz, pz_given_x)
        return kl_div


class HMMAnalytic(HMMVAE):
    def __init__(self, num_steps, num_states):
        super().__init__(num_steps, num_states)

    def reconstruct_log_prob(self, q_dist, x):
        """encoder_dist is tensor of shape [B, T, N], x has shape [B, T, C, Y, X]"""
        log_prob = torch.zeros([x.shape[0]], device=self.device())
        for t in range(x.shape[1]):
            for s in range(self.num_states):
                one_hot = torch.nn.functional.one_hot(torch.tensor(s), self.num_states).float()
                log_prob += torch.exp(q_dist[:, t, s]) * self.observation_model(one_hot).log_prob(x[:, t])
        return log_prob


class HMMMultiSample(HMMVAE):
    def __init__(self, num_steps, num_states, num_z_samples):
        super().__init__(num_steps, num_states)
        self.num_z_samples = num_z_samples

    def reconstruct_log_prob(self, q_dist, x):
        log_probs = [self.sample_reconstruct_log_prob(q_dist, x) for _ in range(self.num_z_samples)]
        log_prob = torch.mean(torch.stack(log_probs), dim=0)
        return log_prob


class HMMUniform(HMMMultiSample):
    def __init__(self, num_steps, num_states, num_z_samples):
        super().__init__(num_steps, num_states, num_z_samples)

    def sample_reconstruct_log_prob(self, q_dist, x):  # Using uniform sampling
        batch_size = q_dist.shape[0]
        log_prob = torch.zeros([x.shape[0]], device=self.device())
        for t in range(x.shape[1]):
            uniform_dist = torch.distributions.one_hot_categorical.OneHotCategorical(
                logits=torch.zeros([batch_size, self.num_states]))
            z = uniform_dist.sample().float()
            q_probs = uniform_dist.log_prob(z).exp()
            log_prob_t = self.observation_model(z).log_prob(x[:, t])
            log_prob += self.num_states * q_probs * log_prob_t
        return log_prob


class HMMReinforce(HMMMultiSample):
    def __init__(self, num_steps, num_states, num_z_samples):
        super().__init__(num_steps, num_states, num_z_samples)

    def sample_reconstruct_log_prob(self, q_dist, x):
        log_prob = torch.zeros([x.shape[0]], device=self.device())
        for t in range(x.shape[1]):
            q_distribution = torch.distributions.one_hot_categorical.OneHotCategorical(logits=q_dist[:, t])
            z = q_distribution.sample()
            q_logits = q_distribution.log_prob(z)
            log_prob_t = self.observation_model(z).log_prob(x[:, t])
            reinforce = log_prob_t.detach() * q_logits
            log_prob += log_prob_t + reinforce - reinforce.detach()
        return log_prob


class HMMReinforceBaseline(HMMVAE):
    def __init__(self, num_steps, num_states, num_z_samples):
        super().__init__(num_steps, num_states)
        net = pixelcnn_dist.make_simple_pixelcnn_net()
        self.baseline_dist = pixelcnn_dist.make_pixelcnn(
            pixelcnn_dist.make_bernoulli_base_distribution(), net, event_shape=[1, 28, 28])
        self.num_z_samples = num_z_samples

    def reconstruct_log_prob(self, q_dist, x):
        baseline_log_prob = torch.zeros([x.shape[0], x.shape[1]], device=self.device())
        for t in range(x.shape[1]):
            baseline_log_prob[:, t] = self.baseline_dist.log_prob(x[:, t])
        log_probs = [self.sample_reconstruct_log_prob(q_dist, x, baseline_log_prob) for _ in range(self.num_z_samples)]
        log_prob = torch.mean(torch.stack(log_probs), dim=0)
        sum_baseline_log_prob = torch.sum(baseline_log_prob, axis=1)
        return log_prob + sum_baseline_log_prob - sum_baseline_log_prob.detach()

    def sample_reconstruct_log_prob(self, q_dist, x, baseline_log_prob):
        log_prob = torch.zeros([x.shape[0]], device=self.device())
        for t in range(x.shape[1]):
            q_distribution = torch.distributions.one_hot_categorical.OneHotCategorical(logits=q_dist[:, t])
            z = q_distribution.sample()
            q_logits = q_distribution.log_prob(z)
            log_prob_t = self.observation_model(z).log_prob(x[:, t])
            reinforce = (log_prob_t - baseline_log_prob[:, t]).detach() * q_logits
            log_prob += log_prob_t + reinforce - reinforce.detach()
        return log_prob


parser = argparse.ArgumentParser(description='PyGen MNIST Sequence HMM')
parser.add_argument("--datasets_folder", default="~/datasets")
parser.add_argument("--tb_folder", default=None)
parser.add_argument("--device", default="cpu")
parser.add_argument("--max_epoch", default=10, type=int)
parser.add_argument("--mode", default="analytic")
parser.add_argument("--num_states", default=10, type=int)
parser.add_argument("--num_z_samples", default=10, type=int)
parser.add_argument("--dummy_run", action="store_true")
ns = parser.parse_args()

num_steps = 3
torch.set_default_device(ns.device)
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), lambda x: (x > 0.5).float(),
    train.DevicePlacement()])
mnist_dataset = torchvision.datasets.MNIST(
    ns.datasets_folder, train=True, download=False,
    transform=transform)
train_mnist_dataset, validation_mnist_dataset = random_split(mnist_dataset, [55000, 5000],
    generator=torch.Generator(device=torch.get_default_device()))
train_dataset = sequential_mnist.SequentialMNISTDataset(train_mnist_dataset, ns.dummy_run)
validation_dataset = sequential_mnist.SequentialMNISTDataset(validation_mnist_dataset, ns.dummy_run)

match ns.mode:
    case "analytic":
        mnist_hmm = HMMAnalytic(num_steps, ns.num_states)
    case "uniform":
        mnist_hmm = HMMUniform(num_steps, ns.num_states, ns.num_z_samples)
    case "reinforce":
        mnist_hmm = HMMReinforce(num_steps, ns.num_states, ns.num_z_samples)
    case "reinforce_baseline":
        mnist_hmm = HMMReinforceBaseline(num_steps, ns.num_states, ns.num_z_samples)
    case _:
        raise RuntimeError(f"mode {ns.mode} not recognised.")

tb_writer = SummaryWriter(ns.tb_folder)
epoch_end_callbacks = callbacks.callback_compose([
    callbacks.tb_conditional_images(tb_writer, "z_conditioned_images", num_labels=ns.num_states),
    callbacks.tb_epoch_log_metrics(tb_writer),
    pygen_models_callbacks.TBSequenceImageCallback(tb_writer, tb_name="image_sequence"),
    pygen_models_callbacks.TBSequenceTransitionMatrixCallback(tb_writer, tb_name="state_transition"),
    callbacks.tb_dataset_metrics_logging(tb_writer, "validation", validation_dataset)
])
train.train(mnist_hmm, train_dataset, pygen_models_train.vae_objective(False),
    batch_end_callback=callbacks.tb_batch_log_metrics(tb_writer),
    epoch_end_callback=epoch_end_callbacks, dummy_run=ns.dummy_run, epoch_regularizer=False)
