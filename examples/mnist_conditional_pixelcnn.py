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
import pygen_models.train.train as pygen_models_train


parser = argparse.ArgumentParser(description='PyGen MNIST PixelCNN')
parser.add_argument("--datasets_folder", default="~/datasets")
parser.add_argument("--tb_folder", default=None)
parser.add_argument("--device", default="cpu")
parser.add_argument("--dummy_run", action="store_true")
parser.add_argument("--num_resnet", default=3, type=int)
ns = parser.parse_args()

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), lambda x: (x > 0.5).float(),
    train.DevicePlacement()])
dataset = torchvision.datasets.MNIST(ns.datasets_folder, train=True, download=False, transform=transform)
train_dataset, validation_dataset = random_split(dataset, [50000, 10000])
torch.set_default_device(ns.device)
tb_writer = SummaryWriter(ns.tb_folder)
epoch_end_callbacks = callbacks.callback_compose([
    callbacks.TBConditionalImages(tb_writer, "conditional_generated_images", num_labels=10),
    callbacks.TBEpochLogMetrics(tb_writer),
    callbacks.TBDatasetMetricsLogging(tb_writer, "validation", validation_dataset)])
conditional_digit_distribution = pixelcnn.PixelCNNQuantizedDistribution([1, 28, 28], 10, ns.num_resnet)
train.train(conditional_digit_distribution, train_dataset, train.OneHotLayerTrainer(10),
    batch_end_callback=callbacks.TBBatchLogMetrics(tb_writer),
    epoch_end_callback=epoch_end_callbacks, dummy_run=ns.dummy_run)
