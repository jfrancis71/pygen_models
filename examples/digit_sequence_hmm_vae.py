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
import pygen.layers.independent_categorical as layer_categorical
import pygen_models.distributions.pixelcnn as pixelcnn_dist
import pygen_models.distributions.hmm as pygen_hmm
from pygen_models.datasets import sequential_mnist
import pygen_models.train.train as pygen_models_train
import pygen_models.train.callbacks as pygen_models_callbacks
import pygen_models.utils.nn_thread as nn_thread
from pygen_models.distributions.discrete_vae import DiscreteVAE


class ObservationDistribution(nn.Module):
    def __init__(self, base_layer, params):
        super().__init__()
        self.base_layer = base_layer
        self.params = params

    def log_prob(self, x):
        return self.base_layer(self.params).log_prob(x).sum(axis=1)

    def sample(self):
        return self.base_layer(self.params).sample()


class ObservationLayer(nn.Module):
    def __init__(self):
        super().__init__()
        conditional_sp_distribution = pixelcnn_layer.make_pixelcnn_layer(
            pixelcnn_dist.make_bernoulli_base_distribution(),
            pixelcnn_layer.make_simple_pixelcnn_net(), [1, 28, 28], ns.num_states)
        self.layer_pixelcnn_bernoulli = nn.Sequential(pixelcnn_layer.SpatialExpand(ns.num_states, ns.num_states,
                                                                              [28, 28]), conditional_sp_distribution)

    def forward(self, params):
        return ObservationDistribution(self.layer_pixelcnn_bernoulli, params)


parser = argparse.ArgumentParser(description='PyGen MNIST Sequence HMM')
parser.add_argument("--datasets_folder", default="~/datasets")
parser.add_argument("--tb_folder", default=None)
parser.add_argument("--device", default="cpu")
parser.add_argument("--max_epoch", default=10, type=int)
parser.add_argument("--num_states", default=10, type=int)
parser.add_argument("--dummy_run", action="store_true")
ns = parser.parse_args()

num_steps = 5
torch.set_default_device(ns.device)
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), lambda x: (x > 0.5).float(),
    train.DevicePlacement()])
mnist_dataset = torchvision.datasets.MNIST(
    ns.datasets_folder, train=True, download=False,
    transform=transform)
train_mnist_dataset, validation_mnist_dataset = random_split(mnist_dataset, [55000, 5000],
    generator=torch.Generator(device=torch.get_default_device()))
train_dataset = sequential_mnist.SequentialMNISTDataset(train_mnist_dataset, num_steps, ns.dummy_run)
validation_dataset = sequential_mnist.SequentialMNISTDataset(validation_mnist_dataset, num_steps, ns.dummy_run)

q = nn.Sequential(nn_thread.NNThread(classifier_net.ClassifierNet(mnist=True, num_classes=ns.num_states), 2),
            nn.Flatten(),
            layer_categorical.IndependentCategorical(event_shape=[num_steps], num_classes=ns.num_states))
mnist_hmm = DiscreteVAE(pygen_hmm.HMM(num_steps, ns.num_states, ObservationLayer()), q)

tb_writer = SummaryWriter(ns.tb_folder)
epoch_end_callbacks = callbacks.callback_compose([
#    callbacks.tb_log_image(tb_writer, "conditional_generated_images",
#                           callbacks.demo_conditional_images(mnist_hmm, torch.eye(ns.num_states),
#                                                             num_samples=2)),
    callbacks.tb_epoch_log_metrics(tb_writer),
    pygen_models_callbacks.TBSequenceImageCallback(tb_writer, tb_name="image_sequence"),
    pygen_models_callbacks.TBSequenceTransitionMatrixCallback(tb_writer, tb_name="state_transition"),
    callbacks.tb_dataset_metrics_logging(tb_writer, "validation", validation_dataset)
])
train.train(mnist_hmm, train_dataset, pygen_models_train.vae_objective(),
    batch_end_callback=callbacks.tb_batch_log_metrics(tb_writer),
    epoch_end_callback=epoch_end_callbacks, dummy_run=ns.dummy_run, max_epoch=ns.max_epoch, epoch_regularizer=False)
