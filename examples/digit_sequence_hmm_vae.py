import argparse
import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.utils import make_grid
import pygen.train.train as train
import pygen.train.callbacks as callbacks
from pygen.neural_nets import classifier_net
import pygen.layers.independent_categorical as layer_categorical
import pygen.layers.independent_bernoulli as bernoulli_layer
import pygen_models.layers.pixelcnn as pixelcnn_layer
import pygen_models.distributions.hmm as pygen_hmm
from pygen_models.datasets import sequential_mnist
import pygen_models.train.train as pygen_models_train
import pygen_models.train.callbacks as pygen_models_callbacks
import pygen_models.utils.nn_thread as nn_thread
from pygen_models.distributions.discrete_vae import DiscreteVAE
from pygen_models.neural_nets import simple_pixelcnn_net


def vae_reconstruct(vae, image_seq):
    def _fn():
        z = vae.q_z_given_x(image_seq.unsqueeze(0)).sample()
        one_hot_z = nn.functional.one_hot(z, vae.latent_model.num_states).float()
        flatten_z = one_hot_z.flatten(-1)
        reconstruct_images = vae.latent_model.p_x_given_z(flatten_z[0]).sample()
        imglist = torch.cat([image_seq, reconstruct_images], dim=0)
        grid_image = make_grid(imglist, padding=10, nrow=5)
        return grid_image
    return _fn


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
channel_layer = bernoulli_layer.IndependentBernoulli(event_shape=[1])
net = simple_pixelcnn_net.SimplePixelCNNNet(1, channel_layer.params_size(), ns.num_states)
layer_pixelcnn_bernoulli = nn.Sequential(
    pixelcnn_layer.SpatialExpand(ns.num_states, ns.num_states,[28, 28]),
    pixelcnn_layer.PixelCNN(net, [1, 28, 28], channel_layer, ns.num_states))
mnist_hmm = DiscreteVAE(pygen_hmm.HMM(num_steps, ns.num_states, layer_pixelcnn_bernoulli), q)

example_valid_images = next(iter(torch.utils.data.DataLoader(validation_dataset, batch_size=10)))[0]
tb_writer = SummaryWriter(ns.tb_folder)
epoch_end_callbacks = callbacks.callback_compose([
    callbacks.tb_log_image(tb_writer, "reconstructed_image_seq", vae_reconstruct(mnist_hmm, example_valid_images[0])),
    callbacks.tb_epoch_log_metrics(tb_writer),
    pygen_models_callbacks.TBSequenceImageCallback(tb_writer, tb_name="image_sequence"),
    pygen_models_callbacks.TBSequenceTransitionMatrixCallback(tb_writer, tb_name="state_transition"),
    callbacks.tb_dataset_metrics_logging(tb_writer, "validation", validation_dataset)
])
train.train(mnist_hmm, train_dataset, pygen_models_train.vae_objective(),
    batch_end_callback=callbacks.tb_batch_log_metrics(tb_writer),
    epoch_end_callback=epoch_end_callbacks, dummy_run=ns.dummy_run, max_epoch=ns.max_epoch, epoch_regularizer=False)
