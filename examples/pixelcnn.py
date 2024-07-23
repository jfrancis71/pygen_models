import argparse
import torch
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pygen.train.train as train
import pygen.train.callbacks as callbacks
import pygen_models.distributions.pixelcnn as pixelcnn
import pygen_models.train.train as pygen_models_train
import pygen_models.train.callbacks as pygen_models_callbacks


parser = argparse.ArgumentParser(description='PyGen CIFAR10 PixelCNN')
parser.add_argument("--datasets_folder", default="~/datasets")
parser.add_argument("--dataset", default="mnist")
parser.add_argument("--tb_folder", default=None)
parser.add_argument("--device", default="cpu")
parser.add_argument("--num_resnet", default=3, type=int)
parser.add_argument("--net", default="simple_pixelcnn_net")
parser.add_argument("--dummy_run", action="store_true")
ns = parser.parse_args()

torch.set_default_device(ns.device)
match ns.net:
    case "simple_pixelcnn_net": net = pixelcnn.make_simple_pixelcnn_net()
    case "pixelcnn_net":  net = pixelcnn.make_pixelcnn_net(ns.num_resnet)
    case _: raise RuntimeError("{ns.net} net name not recognized.")
match ns.dataset:
    case "mnist":
        transform = transforms.Compose([transforms.ToTensor(),
            lambda x: (x > 0.5).float(), train.DevicePlacement()])
        dataset = datasets.MNIST(ns.datasets_folder, train=True, download=False, transform=transform)
        data_split = [55000, 5000]
        event_shape = [1, 28, 28]
        image_distribution = pixelcnn.make_pixelcnn(
            pixelcnn.make_bernoulli_base_distribution(), net, event_shape=[1, 28, 28])
    case "cifar10":
        transform = transforms.Compose([transforms.ToTensor(), train.DevicePlacement()])
        dataset = datasets.CIFAR10(ns.datasets_folder, train=True, download=False, transform=transform)
        data_split = [45000, 5000]
        event_shape = [3, 32, 32]
        image_distribution = pixelcnn.make_pixelcnn(
            pixelcnn.make_quantized_base_distribution(), net, event_shape=[3, 32, 32])
    case _:
        raise RuntimeError("dataset {ns.dataset} not recognized.")
train_dataset, validation_dataset = random_split(dataset, data_split,
    generator=torch.Generator(device=torch.get_default_device()))
tb_writer = SummaryWriter(ns.tb_folder)
epoch_end_callbacks = callbacks.callback_compose([
    pygen_models_callbacks.tb_sample_images(tb_writer, "generated_images"),
    callbacks.tb_epoch_log_metrics(tb_writer),
    callbacks.tb_dataset_metrics_logging(tb_writer, "validation", validation_dataset)
])
train.train(image_distribution, train_dataset, pygen_models_train.distribution_objective,
    batch_end_callback=callbacks.tb_batch_log_metrics(tb_writer),
    epoch_end_callback=epoch_end_callbacks, dummy_run=ns.dummy_run)
