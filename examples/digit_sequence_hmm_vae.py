import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical
import torchvision
import pygen.layers.independent_bernoulli as bernoulli_layer
import pygen.train.callbacks as callbacks
from pygen.neural_nets import classifier_net
import pygen_models.distributions.hmm as hmm
import pygen_models.layers.pixelcnn as pixelcnn_layer
import pygen_models.distributions.pixelcnn as pixelcnn_dist
import pygen_models.distributions.hmm as pygen_hmm
from pygen_models.neural_nets import simple_pixel_cnn_net
from pygen_models.datasets import sequential_mnist
from pygen_models.train.train import VAETrainer
import pygen_models.train.callbacks as pygen_models_callbacks


class LayerPixelCNN(pixelcnn_layer._PixelCNNDistribution):
    def __init__(self, num_conditional):
        pixelcnn_net = simple_pixel_cnn_net.SimplePixelCNNNetwork(num_conditional)
        base_layer = bernoulli_layer.IndependentBernoulli(event_shape=[1])
        super().__init__(pixelcnn_net, [1, 28, 28], base_layer, num_conditional)


class HMMVAE(pygen_hmm.HMM):
    def __init__(self, num_states):
        observation_model = LayerPixelCNN(ns.num_states)
        super().__init__(num_states, observation_model)
        self.q = classifier_net.ClassifierNet(mnist=True, num_classes=self.num_states)

    def log_prob(self, x):
        return self.elbo(x)

    def elbo(self, x):
        q_dist = torch.zeros([x.shape[0], x.shape[1], self.num_states], device=self.device())
        for t in range(x.shape[1]):
            q_dist[:, t] = self.q(x[:, t]).logits
        reconstruct_log_prob = self.reconstruct_log_prob(q_dist, x)
        kl_div = self.kl_div(q_dist)
        return reconstruct_log_prob - kl_div, reconstruct_log_prob, kl_div

    def kl_div(self, q_dist):
        kl_div = self.kl_div_cat(Categorical(logits=q_dist[:, 0]), self.prior_state_distribution())
        for t in range(1, q_dist.shape[1]):
            for s in range(self.num_states):
                kl_div += torch.exp(q_dist[:, t - 1, s]) * self.kl_div_cat(Categorical(logits=q_dist[:, t]), Categorical(logits=self.state_transition_distribution().logits[s]))
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
    def __init__(self, num_states):
        super().__init__(num_states)

    def reconstruct_log_prob(self, q_dist, x):
        """encoder_dist is tensor of shape [B, T, N], x has shape [B, T, C, Y, X]"""
        log_prob = torch.zeros([x.shape[0]], device=self.device())
        for t in range(x.shape[1]):
            for s in range(self.num_states):
                one_hot = torch.nn.functional.one_hot(torch.tensor(s).to(self.device()), self.num_states).float()
                log_prob += torch.exp(q_dist[:, t, s]) * self.observation_model(one_hot).log_prob(x[:, t])
        return log_prob


class HMMMultiSample(HMMVAE):
    def __init__(self, num_states, num_z_samples):
        super().__init__(num_states)
        self.num_z_samples = num_z_samples

    def reconstruct_log_prob(self, q_dist, x):
        log_probs = [self.sample_reconstruct_log_prob(q_dist, x) for _ in range(self.num_z_samples)]
        log_prob = torch.mean(torch.stack(log_probs), dim=0)
        return log_prob


class HMMUniform(HMMMultiSample):
    def __init__(self, num_states, num_z_samples):
        super().__init__(num_states, num_z_samples)

    def sample_reconstruct_log_prob(self, q_dist, x):  # Using uniform sampling
        batch_size = q_dist.shape[0]
        log_prob = torch.zeros([x.shape[0]], device=self.device())
        for t in range(x.shape[1]):
            z = torch.distributions.categorical.Categorical(logits=torch.zeros([batch_size, self.num_states]).to(self.device())).sample()
            one_hot = torch.nn.functional.one_hot(z, self.num_states).float()
            q_probs = torch.exp(q_dist[:,t])[torch.arange(batch_size), z]
            log_prob_t = self.observation_model(one_hot).log_prob(x[:, t])
            log_prob += self.num_states * q_probs * log_prob_t
        return log_prob


class HMMReinforce(HMMMultiSample):
    def __init__(self, num_states, num_z_samples):
        super().__init__(num_states, num_z_samples)

    def sample_reconstruct_log_prob(self, q_dist, x):
        batch_size = q_dist.shape[0]
        log_prob = torch.zeros([x.shape[0]], device=self.device())
        for t in range(x.shape[1]):
            z = torch.distributions.categorical.Categorical(logits=q_dist[:,t]).sample()
            one_hot = torch.nn.functional.one_hot(z, self.num_states).float()
            q_logits = q_dist[:,t][torch.arange(batch_size), z]
            log_prob_t = self.observation_model(one_hot).log_prob(x[:, t])
            reinforce = log_prob_t.detach() * q_logits
            log_prob += log_prob_t + reinforce - reinforce.detach()
        return log_prob


class HMMReinforceBaseline(HMMMultiSample):
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
        baseline_log_prob = torch.zeros([x.shape[0], x.shape[1]], device=self.device())
        for t in range(x.shape[1]):
            baseline_log_prob[:, t] = self.baseline_dist.log_prob(x[:, t])
        log_probs = [self.sample_reconstruct_log_prob(q_dist, x, baseline_log_prob) for _ in range(self.num_z_samples)]
        log_prob = torch.mean(torch.stack(log_probs), dim=0)
        sum_baseline_log_prob = torch.sum(baseline_log_prob, axis=1)
        return log_prob + sum_baseline_log_prob - sum_baseline_log_prob.detach()

    def sample_reconstruct_log_prob(self, q_dist, x, baseline_log_prob):
        batch_size = q_dist.shape[0]
        log_prob = torch.zeros([x.shape[0]], device=self.device())
        for t in range(x.shape[1]):
            z = torch.distributions.categorical.Categorical(logits=q_dist[:,t]).sample()
            one_hot = torch.nn.functional.one_hot(z, self.num_states).float()
            q_logits = q_dist[:,t][torch.arange(batch_size), z]
            log_prob_t = self.observation_model(one_hot).log_prob(x[:, t])
            reinforce = (log_prob_t - baseline_log_prob[:, t]).detach() * q_logits
            log_prob += log_prob_t + reinforce - reinforce.detach()
        return log_prob


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

match ns.mode:
    case "analytic":
        mnist_hmm = HMMAnalytic(ns.num_states)
    case "uniform":
        mnist_hmm = HMMUniform(ns.num_states, ns.num_z_samples)
    case "reinforce":
        mnist_hmm = HMMReinforce(ns.num_states, ns.num_z_samples)
    case "reinforce_baseline":
        mnist_hmm = HMMReinforceBaseline(ns.num_states, ns.num_z_samples)
    case _:
        raise RuntimeError(f"mode {ns.mode} not recognised.")

tb_writer = SummaryWriter(ns.tb_folder)
batch_end_callbacks = callbacks.callback_compose([
    callbacks.TBBatchLogProbCallback(tb_writer, "batch_log_prob"),
    pygen_models_callbacks.TBBatchVAECallback(tb_writer)
])
epoch_end_callbacks = callbacks.callback_compose([
    callbacks.TBConditionalImagesCallback(tb_writer, "z_conditioned_images", num_labels=ns.num_states),
    callbacks.TBDatasetLogProbCallback(tb_writer, "validation_log_prob", validation_dataset),
    pygen_models_callbacks.TBSequenceImageCallback(tb_writer, tb_name="image_sequence"),
    pygen_models_callbacks.TBSequenceTransitionMatrixCallback(tb_writer, tb_name="state_transition"),
])
trainer = VAETrainer(
    mnist_hmm.to(ns.device),
    train_dataset, max_epoch=ns.max_epoch, batch_end_callback=batch_end_callbacks, epoch_end_callback=epoch_end_callbacks, dummy_run=ns.dummy_run)
torch.autograd.set_detect_anomaly(True)
trainer.train()
