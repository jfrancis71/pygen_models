import argparse

import torch
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pygen.train.train as train
import pygen.train.callbacks as callbacks
import pygen_models.layers.pixelcnn as pixelcnn


class ConditionalDistribution(nn.Module):
    def __init__(self, nr_resnet=3):
        super().__init__()
        self.layer = pixelcnn.PixelCNNQuantizedDistribution([3, 32, 32], 10, nr_resnet)

    def forward(self, x):
        one_hot = F.one_hot(x, num_classes=10)
        return self.layer(one_hot.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 32, 32).type(torch.float))


parser = argparse.ArgumentParser(description='PyGen CIFAR10 Conditional PixelCNN')
parser.add_argument("--datasets_folder", default=".")
parser.add_argument("--tb_folder", default=None)
parser.add_argument("--device", default="cpu")
parser.add_argument("--dummy_run", action="store_true")
parser.add_argument("--num_resnet", default=3, type=int)
ns = parser.parse_args()

transform = torchvision.transforms.ToTensor()
dataset = torchvision.datasets.CIFAR10(ns.datasets_folder, train=True, download=True, transform=transform)
train_dataset, validation_dataset = random_split(dataset, [45000, 5000])
tb_writer = SummaryWriter(ns.tb_folder)
epoch_end_callback = callbacks.callback_compose([
    callbacks.TBConditionalImagesCallback(tb_writer, "conditional_generated_images", num_labels=10),
    callbacks.TBTotalLogProbCallback(tb_writer, "train_epoch_log_prob"),
    callbacks.TBDatasetLogProbLayerCallback(tb_writer, "validation_log_prob", validation_dataset, reverse_inputs=True)])
conditional_distribution = ConditionalDistribution(ns.num_resnet)
train.LayerTrainer(
    conditional_distribution.to(ns.device),
    train_dataset,
    batch_end_callback=callbacks.TBBatchLogProbCallback(tb_writer, "batch_log_prob"),
    epoch_end_callback=epoch_end_callback, reverse_inputs=True, dummy_run=ns.dummy_run).train()
