import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical
import torchvision
import pygen.train.callbacks as callbacks
from pygen.train import train
import pygen_models.distributions.hmm as hmm
import pygen.layers.independent_bernoulli as bernoulli_layer
import pygen_models.layers.pixelcnn as pixelcnn_layer
import pygen_models.distributions.pixelcnn as pixelcnn_dist
import pygen_models.distributions.hmm as pygen_hmm
from pygen.neural_nets import classifier_net
from pygen_models.neural_nets import simple_pixel_cnn_net
from pygen_models.datasets import sequential_mnist


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


class LayerPixelCNN(pixelcnn_layer._PixelCNNDistribution):
    def __init__(self, num_conditional):
        pixelcnn_net = simple_pixel_cnn_net.SimplePixelCNNNetwork(num_conditional)
        base_layer = bernoulli_layer.IndependentBernoulli(event_shape=[1])
        super().__init__(pixelcnn_net, [1, 28, 28], base_layer, num_conditional)

    def forward(self, x):
        return super().forward(x.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 28, 28))


class HMMVAE(pygen_hmm.HMM):
    def __init__(self, num_states):
        observation_model = LayerPixelCNN(ns.num_states)
        super().__init__(num_states, observation_model)
        self.q = Encoder(num_states)

    def log_prob(self, x):
        return self.elbo(x)

    def elbo(self, x):
        q_dist = torch.zeros([x.shape[0], x.shape[1], self.num_states], device=self.device())
        for t in range(x.shape[1]):
            q_dist[:, t] = self.q(x[:, t]).logits
        log_prob = self.reconstruct_log_prob(q_dist, x)
        kl_div = self.kl_div(q_dist)
        return log_prob - kl_div

    def kl_div(self, q_dist):
        kl_div = self.kl_div_cat(Categorical(logits=q_dist[:, 0]), self.prior_state_distribution())
        for t in range(1, q_dist.shape[1]):
            for s in range(self.num_states):
                kl_div += torch.exp(q_dist[:, t - 1, s]) * self.kl_div_cat(Categorical(logits=q_dist[:, t]), Categorical(logits=self.state_transition_distribution().logits[s]))
        return kl_div

    def kl_div_cat(self, p, q):
        kl_div = torch.sum(p.probs * (p.logits - q.logits), axis=1)
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
        raise RuntimeError("mode {} not recognised.".format(ns.mode))

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
