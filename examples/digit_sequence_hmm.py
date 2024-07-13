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
import pygen_models.layers.pixelcnn as pixelcnn_layer
import torch.nn.functional as F
from pygen_models.datasets import sequential_mnist
from pygen_models.neural_nets import simple_pixel_cnn_net
from pygen_models.train.train import RegTrainer
import pygen_models.train.callbacks as pygen_models_callbacks


class LayerPixelCNN(pixelcnn_layer._PixelCNNDistribution):
    def __init__(self, num_conditional):
        pixelcnn_net = simple_pixel_cnn_net.SimplePixelCNNNetwork(num_conditional)
        base_layer = bernoulli_layer.IndependentBernoulli(event_shape=[1])
        super().__init__(pixelcnn_net, [1, 28, 28], base_layer, num_conditional)


class HMMReg(hmm.HMMAnalytic):
    def __init__(self, num_states, observation_model):
        super().__init__(num_states, observation_model)

    def reg(self, observations):  # x is B*Event_shape, ie just one element of sequence
        emission_logits = self.emission_logits(observations[:, 0])
        pz = Categorical(logits=emission_logits*0.0)
        pz_given_x = Categorical(logits=emission_logits)
        kl_div = torch.sum(pz.probs * (pz.logits - pz_given_x.logits), axis=1)
        return kl_div


parser = argparse.ArgumentParser(description='PyGen MNIST Sequence HMM')
parser.add_argument("--datasets_folder", default=".")
parser.add_argument("--tb_folder", default=None)
parser.add_argument("--device", default="cpu")
parser.add_argument("--max_epoch", default=10, type=int)
parser.add_argument("--num_states", default=10, type=int)
parser.add_argument("--dummy_run", action="store_true")
ns = parser.parse_args()

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), lambda x: (x > 0.5).float()])
mnist_dataset = torchvision.datasets.MNIST(
    ns.datasets_folder, train=True, download=False,
    transform=transform)
train_mnist_dataset, validation_mnist_dataset = random_split(mnist_dataset, [55000, 5000])
train_dataset = sequential_mnist.SequentialMNISTDataset(train_mnist_dataset, ns.dummy_run)
validation_dataset = sequential_mnist.SequentialMNISTDataset(validation_mnist_dataset, ns.dummy_run)

layer_pixelcnn_bernoulli = LayerPixelCNN(ns.num_states)
mnist_hmm = HMMReg(ns.num_states, layer_pixelcnn_bernoulli)

tb_writer = SummaryWriter(ns.tb_folder)
epoch_end_callbacks = callbacks.callback_compose([
    callbacks.TBConditionalImagesCallback(tb_writer, "z_conditioned_images", num_labels=ns.num_states),
    callbacks.TBDatasetLogProbCallback(tb_writer, "validation_log_prob", validation_dataset),
    pygen_models_callbacks.TBSequenceImageCallback(tb_writer, tb_name="image_sequence"),
    pygen_models_callbacks.TBSequenceTransitionMatrixCallback(tb_writer, tb_name="state_transition"),
])
trainer = RegTrainer(
    mnist_hmm.to(ns.device),
    train_dataset, max_epoch=ns.max_epoch,
    batch_end_callback=callbacks.TBBatchLogProbCallback(tb_writer, "batch_log_prob"),
    epoch_end_callback=epoch_end_callbacks, dummy_run=ns.dummy_run)
trainer.train()
