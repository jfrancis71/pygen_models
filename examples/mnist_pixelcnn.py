import argparse
import torch
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision
import pygen.train.train as train
import pygen.train.callbacks as callbacks
import pygen_models.distributions.pixelcnn as pixelcnn
import pygen_models.train.train as pygen_models_train

parser = argparse.ArgumentParser(description='PyGen MNIST PixelCNN')
parser.add_argument("--datasets_folder", default="~/datasets")
parser.add_argument("--tb_folder", default=None)
parser.add_argument("--device", default="cpu")
parser.add_argument("--num_resnet", default=3, type=int)
parser.add_argument("--dummy_run", action="store_true")
ns = parser.parse_args()

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), lambda x: (x > 0.5).float(), train.DevicePlacement()])
dataset = torchvision.datasets.MNIST(ns.datasets_folder, train=True, download=False, transform=transform)
train_dataset, validation_dataset = random_split(dataset, [50000, 10000])
torch.set_default_device(ns.device)
tb_writer = SummaryWriter(ns.tb_folder)
epoch_end_callbacks = callbacks.callback_compose([
    callbacks.TBSampleImages(tb_writer, "generated_images"),
    callbacks.TBEpochLogMetrics(tb_writer),
    callbacks.TBDatasetMetricsLogging(tb_writer, "validation", validation_dataset)
])
digit_distribution = pixelcnn.PixelCNNBernoulliDistribution(event_shape=[1, 28, 28], nr_resnet=ns.num_resnet)
train.train(digit_distribution, train_dataset, pygen_models_train.distribution_trainer,
    batch_end_callback=callbacks.TBBatchLogMetrics(tb_writer),
    epoch_end_callback=epoch_end_callbacks, dummy_run=ns.dummy_run)
