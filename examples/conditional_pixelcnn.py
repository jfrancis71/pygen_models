import argparse
import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pygen.train.train as train
import pygen.train.callbacks as callbacks
import pygen.layers.one_hot as one_hot
import pygen_models.layers.pixelcnn as pixelcnn
import pygen_models.distributions.pixelcnn as pixelcnn_dist


parser = argparse.ArgumentParser(description='PyGen Conditional PixelCNN')
parser.add_argument("--datasets_folder", default="~/datasets")
parser.add_argument("--dataset", default="mnist")
parser.add_argument("--tb_folder", default=None)
parser.add_argument("--device", default="cpu")
parser.add_argument("--net", default="simple_pixelcnn_net")
parser.add_argument("--num_resnet", default=3, type=int)
parser.add_argument("--max_epoch", default=10, type=int)
parser.add_argument("--dummy_run", action="store_true")
ns = parser.parse_args()

torch.set_default_device(ns.device)
match ns.net:
    case "simple_pixelcnn_net": net = pixelcnn.make_simple_pixelcnn_net()
    case "pixelcnn_net":  net = pixelcnn.make_pixelcnn_net(ns.num_resnet)
    case _: raise RuntimeError("{ns.net} net name not recognized.")
if ns.dataset == "mnist":
    transform = transforms.Compose([transforms.ToTensor(),
        lambda x: (x > 0.5).float(), train.DevicePlacement()])
    dataset = datasets.MNIST(ns.datasets_folder, train=True, download=False, transform=transform)
    event_shape = [1, 28, 28]
    data_split = [55000, 5000]
    conditional_sp_distribution = pixelcnn.make_pixelcnn_layer(pixelcnn_dist.make_bernoulli_base_distribution(),
        net, event_shape, 3)
elif ns.dataset == "cifar10":
    transform = transforms.Compose([transforms.ToTensor(), train.DevicePlacement()])
    dataset = datasets.CIFAR10(ns.datasets_folder, train=True, download=False, transform=transform)
    event_shape = [3, 32, 32]
    data_split = [45000, 5000]
    conditional_sp_distribution = pixelcnn.make_pixelcnn_layer(pixelcnn_dist.make_quantized_base_distribution(),
        net, event_shape, 3)
else:
    raise RuntimeError(f"{ns.dataset} not recognized.")
train_dataset, validation_dataset = random_split(dataset, data_split,
    generator=torch.Generator(device=torch.get_default_device()))
tb_writer = SummaryWriter(ns.tb_folder)
conditional_distribution = nn.Sequential(
    one_hot.OneHot(10),
    pixelcnn.SpatialExpand(10, 3, event_shape[1:]),
    conditional_sp_distribution)
epoch_end_callback = callbacks.callback_compose([
    callbacks.tb_log_image(tb_writer, "conditional_generated_images",
        callbacks.demo_conditional_images(conditional_distribution, torch.arange(10), num_samples=2)),
    callbacks.tb_epoch_log_metrics(tb_writer),
    callbacks.tb_dataset_metrics_logging(tb_writer, "validation", validation_dataset)])
train.train(
    conditional_distribution, train_dataset, train.layer_objective(reverse_inputs=True),
    batch_end_callback=callbacks.tb_batch_log_metrics(tb_writer),
    epoch_end_callback=epoch_end_callback, dummy_run=ns.dummy_run, max_epoch=ns.max_epoch)
