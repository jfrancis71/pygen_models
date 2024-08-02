import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision
import pygen.train.callbacks as callbacks
from pygen.neural_nets import classifier_net
import pygen.layers.categorical as layer_categorical
import pygen.train.train as train
import pygen_models.layers.pixelcnn as pixelcnn_layer
import pygen_models.distributions.pixelcnn as pixelcnn_dist

class OneHot(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        return nn.functional.one_hot(x, self.num_classes).float()

class MutualInformation(nn.Module):
    def __init__(self, num_states):
        super().__init__()
        self.num_states = num_states
        self.classifier = nn.Sequential(classifier_net.ClassifierNet(mnist=True, num_classes=self.num_states), layer_categorical.Categorical())
        intermediate_channels = 3
        net = pixelcnn_layer.make_simple_pixelcnn_net()
        conditional_sp_distribution = pixelcnn_layer.make_pixelcnn_layer(
            pixelcnn_dist.make_bernoulli_base_distribution(), net, [1, 28, 28], intermediate_channels)
        self.p_x_given_z = nn.Sequential(OneHot(self.num_states), pixelcnn_layer.SpatialExpand(self.num_states, intermediate_channels, [28, 28]),
                                         conditional_sp_distribution)

    def mutual_info(self, x):
        pz_given_x = self.classifier(x)
        entropy_z_given_x = pz_given_x.entropy()
        mean_z = pz_given_x.probs.mean(axis=0)
        entropy_z = torch.distributions.categorical.Categorical(probs=mean_z).entropy()
        mutual_information = entropy_z - entropy_z_given_x.mean()
        sample_z = pz_given_x.sample()
        logits_px_given_z = self.p_x_given_z(sample_z).log_prob(x).mean()
        grad = logits_px_given_z - logits_px_given_z.detach()  # get a gradient to train p(x|z)
        return mutual_information + grad

    def forward(self, z):
        return self.p_x_given_z(z.max(axis=1)[1])


def mi_objective(trainable, batch):
    mi = (trainable.mutual_info(batch[0]))
    return mi, np.array((mi.cpu().detach().numpy()), dtype=[('mutual_information', 'float32')])


parser = argparse.ArgumentParser(description='PyGen MNIST Mutual Information')
parser.add_argument("--datasets_folder", default="~/datasets")
parser.add_argument("--tb_folder", default=None)
parser.add_argument("--device", default="cpu")
parser.add_argument("--max_epoch", default=10, type=int)
parser.add_argument("--dummy_run", action="store_true")
parser.add_argument("--num_states", default=10, type=int)
ns = parser.parse_args()

torch.set_default_device(ns.device)
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), lambda x: (x > 0.5).float(), train.DevicePlacement()])
mnist_dataset = torchvision.datasets.MNIST(
    ns.datasets_folder, train=True, download=False,
    transform=transform)
train_dataset, validation_dataset = random_split(mnist_dataset, [50000, 10000], generator=torch.Generator(device=torch.get_default_device()))
tb_writer = SummaryWriter(ns.tb_folder)
digit_mi = MutualInformation(ns.num_states)
epoch_callbacks = callbacks.tb_conditional_images(tb_writer, "z_conditioned_images", num_labels=ns.num_states, num_samples=5)
train.train(digit_mi, train_dataset, mi_objective,
    batch_end_callback=callbacks.tb_batch_log_metrics(tb_writer), epoch_end_callback=epoch_callbacks, dummy_run=ns.dummy_run)
