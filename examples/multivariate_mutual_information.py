import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision
from torchvision.utils import make_grid
import pygen.train.callbacks as callbacks
from pygen.neural_nets import classifier_net
import pygen.train.train as train
import pygen_models.layers.pixelcnn as pixelcnn_layer
import pygen_models.distributions.pixelcnn as pixelcnn_dist

class OneHot(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        return nn.functional.one_hot(x, self.num_classes).float()

class IndependentCategorical(nn.Module):
    def __init__(self, num_vars, num_states):
        super().__init__()
        self.num_vars = num_vars
        self.num_states = num_states

    def forward(self, x):
        batch_shape = x.shape[:-1]
        reshape_logits = x.reshape(batch_shape + torch.Size([self.num_vars, self.num_states]))
        categorical = torch.distributions.categorical.Categorical(logits=reshape_logits)
        return torch.distributions.independent.Independent(base_distribution=categorical, reinterpreted_batch_ndims=1)


class MutualInformation(nn.Module):
    def __init__(self, num_states, num_vars):
        super().__init__()
        self.num_states = num_states
        self.num_vars = num_vars
        self.classifier = nn.Sequential(classifier_net.ClassifierNet(mnist=True, num_classes=self.num_states*self.num_vars), IndependentCategorical(num_vars, num_states))
        intermediate_channels = 16
        #net = pixelcnn_layer.make_simple_pixelcnn_net()
        net = pixelcnn_layer.make_pixelcnn_net(1)
        conditional_sp_distribution = pixelcnn_layer.make_pixelcnn_layer(
            pixelcnn_dist.make_bernoulli_base_distribution(), net, [1, 28, 28], intermediate_channels)
        self.p_x_given_z = nn.Sequential(pixelcnn_layer.SpatialExpand(self.num_states*self.num_vars, intermediate_channels, [28, 28]),
                                         conditional_sp_distribution)

    def mutual_info(self, x):
        pz_given_x = self.classifier(x)
        entropy_z_given_x = pz_given_x.entropy()
        base_dist = torch.distributions.categorical.Categorical(pz_given_x.base_dist.probs.mean(axis=0))
        p_z = torch.distributions.independent.Independent(base_dist, reinterpreted_batch_ndims=1)
        entropy_z = p_z.entropy()
        mutual_information = entropy_z - entropy_z_given_x.mean()
        sample_z = pz_given_x.sample()
        reshape_z = torch.reshape(nn.functional.one_hot(sample_z, self.num_states).float(), [x.shape[0],self.num_vars*self.num_states])
        logits_px_given_z = self.p_x_given_z(reshape_z).log_prob(x).mean()
        grad = logits_px_given_z - logits_px_given_z.detach()  # get a gradient to train p(x|z)
        return mutual_information + grad, logits_px_given_z.detach()

    def forward(self, z):
        return self.p_x_given_z(z.max(axis=1)[1])


def mi_objective(trainable, batch):
    mi, log_prob_x = (trainable.mutual_info(batch[0]))
    return mi, np.array(
        (mi.cpu().detach().numpy(), log_prob_x),
        dtype=[('mutual_information', 'float32'), ('log_prob_x_given_z', 'float32')])


def tb_reconstruct_images(tb_writer, images, num_vars, num_states):
    def _fn(trainer_state):
        z_samples = trainer_state.trainable.classifier(images).sample()
        reshape_z = torch.reshape(nn.functional.one_hot(z_samples, num_states).float(),
                                  [16, num_vars * num_states])
        reconstruct_images = trainer_state.trainable.p_x_given_z(reshape_z).sample()
        imglist = torch.cat([images, reconstruct_images], axis=0)
        grid_image = make_grid(imglist, padding=10, nrow=16, value_range=(0.0, 1.0))
        tb_writer.add_image("z_images", grid_image, trainer_state.epoch_num)

    return _fn


def tb_dataset_mi(tb_writer, dataset, batch_size=32):
    def _fn(trainer_state):
        num_vars, num_states = trainer_state.trainable.num_vars, trainer_state.trainable.num_states
        dataloader = DataLoader(dataset, collate_fn=None,
            generator=torch.Generator(device=torch.get_default_device()),
            batch_size=batch_size, shuffle=True, drop_last=True)
        dataset_iter = iter(dataloader)
        list_pz_given_x = []
        for batch in dataset_iter:
            pz_given_x = trainer_state.trainable.classifier(batch[0]).base_dist.logits
            list_pz_given_x.append(pz_given_x)
        pz_given_x_logits = torch.cat(list_pz_given_x)
        pz_given_x = IndependentCategorical(num_vars, num_states)(pz_given_x_logits.flatten(1))
        base_dist = torch.distributions.categorical.Categorical(pz_given_x.base_dist.probs.mean(axis=0))
        p_z = torch.distributions.independent.Independent(base_dist, reinterpreted_batch_ndims=1)
        entropy_z_given_x = pz_given_x.entropy()
        entropy_z = p_z.entropy()
        mutual_information = entropy_z - entropy_z_given_x.mean()
        tb_writer.add_scalar("epoch__mi", mutual_information, trainer_state.epoch_num)
    return _fn


parser = argparse.ArgumentParser(description='PyGen MNIST Mutual Information')
parser.add_argument("--datasets_folder", default="~/datasets")
parser.add_argument("--tb_folder", default=None)
parser.add_argument("--device", default="cpu")
parser.add_argument("--max_epoch", default=10, type=int)
parser.add_argument("--dummy_run", action="store_true")
parser.add_argument("--num_states", default=10, type=int)
parser.add_argument("--num_vars", default=3, type=int)
ns = parser.parse_args()

torch.set_default_device(ns.device)
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), lambda x: (x > 0.5).float(), train.DevicePlacement()])
mnist_dataset = torchvision.datasets.MNIST(
    ns.datasets_folder, train=True, download=False,
    transform=transform)
train_dataset, validation_dataset = random_split(mnist_dataset, [50000, 10000], generator=torch.Generator(device=torch.get_default_device()))
example_valid_images = next(iter(torch.utils.data.DataLoader(validation_dataset, batch_size=16)))[0]
tb_writer = SummaryWriter(ns.tb_folder)
digit_mi = MutualInformation(ns.num_states, ns.num_vars)
epoch_end_callbacks_list = [
    callbacks.tb_epoch_log_metrics(tb_writer),
    tb_reconstruct_images(tb_writer, example_valid_images, ns.num_vars, ns.num_states),
    tb_dataset_mi(tb_writer, validation_dataset)]
epoch_end_callbacks = callbacks.callback_compose(epoch_end_callbacks_list)
train.train(digit_mi, train_dataset, mi_objective,
    batch_end_callback=callbacks.tb_batch_log_metrics(tb_writer), epoch_end_callback=epoch_end_callbacks, dummy_run=ns.dummy_run)
