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


class LayerPixelCNN(pixelcnn_layer._PixelCNNDistribution):
    def __init__(self, num_conditional):
        pixelcnn_net = simple_pixel_cnn_net.SimplePixelCNNNetwork(num_conditional)
        base_layer = bernoulli_layer.IndependentBernoulli(event_shape=[1])
        super().__init__(pixelcnn_net, [1, 28, 28], base_layer, num_conditional)


class HMMTrainer(train.DistributionTrainer):
    # pylint: disable=R0913
    def __init__(self, trainable, dataset, batch_size=32, max_epoch=10, batch_end_callback=None,
                 epoch_end_callback=None, use_scheduler=False, dummy_run=False, model_path=None):
        super().__init__(
            trainable, dataset, batch_size, max_epoch, batch_end_callback,
            epoch_end_callback, use_scheduler=use_scheduler, dummy_run=dummy_run,
            model_path=model_path)

    def kl_div(self, observation):  # x is B*Event_shape, ie just one element of sequence
        emission_logits = self.trainable.emission_logits(observation)
        pz = Categorical(logits=emission_logits*0.0)
        pz_given_x = Categorical(logits=emission_logits)
        kl_div = torch.sum(pz.probs * (pz.logits - pz_given_x.logits), axis=1)
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
mnist_hmm = hmm.HMMAnalytic(ns.num_states, layer_pixelcnn_bernoulli)

tb_writer = SummaryWriter(ns.tb_folder)
epoch_end_callbacks = callbacks.callback_compose([
    callbacks.TBConditionalImagesCallback(tb_writer, "z_conditioned_images", num_labels=ns.num_states),
    callbacks.TBDatasetLogProbCallback(tb_writer, "validation_log_prob", validation_dataset),
    TBSequenceImageCallback(tb_writer, tb_name="image_sequence"),
    TBSequenceTransitionMatrixCallback(tb_writer, tb_name="state_transition"),
])
trainer = HMMTrainer(
    mnist_hmm.to(ns.device),
    train_dataset, max_epoch=ns.max_epoch, epoch_end_callback=epoch_end_callbacks, dummy_run=ns.dummy_run)
trainer.train()
