import argparse
import torch
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pygen.train.train as train
import pygen.train.callbacks as callbacks
import pygen_models.layers.pixelcnn as pixelcnn


parser = argparse.ArgumentParser(description='PyGen Conditional PixelCNN')
parser.add_argument("--datasets_folder", default="~/datasets")
parser.add_argument("--dataset", default="mnist")
parser.add_argument("--tb_folder", default=None)
parser.add_argument("--device", default="cpu")
parser.add_argument("--dummy_run", action="store_true")
parser.add_argument("--num_resnet", default=3, type=int)
ns = parser.parse_args()

if ns.dataset == "mnist":
    transform = transforms.Compose([transforms.ToTensor(),
        lambda x: (x > 0.5).float(), train.DevicePlacement()])
    dataset = datasets.MNIST(ns.datasets_folder, train=True, download=False, transform=transform)
    event_shape = [1, 28, 28]
    data_split = [55000, 5000]
elif ns.dataset == "cifar10":
    transform = transforms.Compose([transforms.ToTensor(), train.DevicePlacement()])
    dataset = datasets.CIFAR10(ns.datasets_folder, train=True, download=False, transform=transform)
    event_shape = [3, 32, 32]
    data_split = [45000, 5000]
else:
    raise RuntimeError(f"{ns.dataset} not recognized.")
train_dataset, validation_dataset = random_split(dataset, data_split)
torch.set_default_device(ns.device)
tb_writer = SummaryWriter(ns.tb_folder)
epoch_end_callback = callbacks.callback_compose([
    callbacks.tb_conditional_images(tb_writer, "conditional_generated_images", num_labels=10),
    callbacks.tb_epoch_log_metrics(tb_writer),
    callbacks.tb_dataset_metrics_logging(tb_writer, "validation", validation_dataset)
])
if ns.dataset == "mnist":
    conditional_distribution = (
        pixelcnn.PixelCNNBernoulliDistribution([1, 28, 28], 10, ns.num_resnet))
elif ns.dataset == "cifar10":
    conditional_distribution = (
        pixelcnn.PixelCNNQuantizedDistribution([3, 32, 32], 10, ns.num_resnet))
train.train(
    conditional_distribution, train_dataset, train.layer_objective(reverse_inputs=True, num_classes=10),
    batch_end_callback=callbacks.tb_batch_log_metrics(tb_writer),
    epoch_end_callback=epoch_end_callback, dummy_run=ns.dummy_run)
