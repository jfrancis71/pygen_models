import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
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


class MNISTSequenceDataset(Dataset):
    def __init__(self, mnist_dataset):
        mnist_dataset_len = len(mnist_dataset)
#        mnist_dataset_len = 100
        mnist_digits = torch.stack([mnist_dataset[n][0] for n in range(mnist_dataset_len)]).float()
        mnist_labels = torch.tensor([mnist_dataset[n][1] for n in range(mnist_dataset_len)])
        self.digits = [mnist_digits[mnist_labels == d] for d in range(10)]

    def __len__(self):
        return 60000

    def __getitem__(self, idx):
        d = torch.randint(low=0, high=9, size=[])
        rand_idx_sel = torch.randint(low=0, high=self.digits[d].shape[0], size=[])
        i1 = [self.digits[d][rand_idx_sel]]
        for s in range(4):
            d = (d + 1) % 10
            rand_idx_sel = torch.randint(low=0, high=self.digits[d].shape[0], size=[])
            i1.append(self.digits[d][rand_idx_sel])
        return (torch.stack(i1),)


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
        self.prj1 = nn.Linear(num_conditional, 32*28*28)
        self.conv2 = MaskedConv2d(32, 1, 1, 1)

    def forward(self, x, sample=False, conditional=None):
        x = self.conv1(x)
        prj = self.prj1(conditional[:,:,0,0]).reshape([-1, 32, 28, 28])
        x = x + prj
        x = F.relu(x)
        x = self.conv2(x)
        return x


class ImageObservationModel(nn.Module):
    def __init__(self, num_states, event_shape, device):
        super().__init__()
        # pylint: disable=E1101
        self.num_states = num_states
        self.device = device
        self.event_shape = event_shape
        self.pixelcnn_net = SimplePixelCNNNetwork(self.num_states)
        self.base_layer = bernoulli_layer.IndependentBernoulli(event_shape=[1])

    def emission_logits(self, observation):  # Batch + event_shape
        """returns vector of length num_states representing log p(observation | state)"""
        emission_probs = torch.stack([self.state_emission_distribution(s).log_prob(observation)
                                      for s in range(self.num_states)]).transpose(0, 1)
        return emission_probs

    def state_emission_distribution(self, s):
        encode = torch.nn.functional.one_hot(torch.tensor(s).to(self.device), self.num_states).float()
        dist = pixelcnn_dist._PixelCNN(
            [1, 28, 28],
            self.base_layer,
            encode.unsqueeze(-1).unsqueeze(-1).repeat(1, 28, 28)
            )
        dist.pixelcnn_net = self.pixelcnn_net
        return dist


class HMMTrainer(train.DistributionTrainer):
    # pylint: disable=R0913
    def __init__(self, trainable, dataset, batch_size=32, max_epoch=10, batch_end_callback=None,
                 epoch_end_callback=None, use_scheduler=False, dummy_run=False, model_path=None):
        super().__init__(
            trainable, dataset, batch_size, max_epoch, batch_end_callback,
            epoch_end_callback, use_scheduler=use_scheduler, dummy_run=dummy_run,
            model_path=model_path)

    def kl_div(self, observation):  # x is B*Event_shape, ie just one element of sequence
        emission_logits = self.trainable.observation_model.emission_logits(observation)
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
dataset = MNISTSequenceDataset(mnist_dataset)
train_dataset, validation_dataset = random_split(dataset, [55000, 5000])

mnist_hmm = hmm.HMM(ns.num_states, ImageObservationModel(ns.num_states, [1, 28, 28], ns.device))

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

trainer.train()
