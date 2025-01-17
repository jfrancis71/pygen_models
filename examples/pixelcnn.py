import argparse
import torch
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pygen.train.train as train
import pygen.train.callbacks as callbacks
import pygen_models.distributions.quantized_distribution as qd
import pygen_models.layers.independent_quantized_distribution as ql
import pygen.layers.independent_bernoulli as bernoulli_layer
import pygen_models.distributions.pixelcnn as pixelcnn
import pygen_models.train.train as pygen_models_train
import pygen_models.train.callbacks as pygen_models_callbacks
from pygen_models.neural_nets import simple_pixelcnn_net
import pixelcnn_pp.model as pixelcnn_net_pp


parser = argparse.ArgumentParser(description='PyGen PixelCNN')
parser.add_argument("--datasets_folder", default="~/datasets")
parser.add_argument("--dataset", default="mnist")
parser.add_argument("--tb_folder", default=None)
parser.add_argument("--images_folder", default=None)
parser.add_argument("--device", default="cpu")
parser.add_argument("--num_resnet", default=3, type=int)
parser.add_argument("--net", default="simple_pixelcnn_net")
parser.add_argument("--dummy_run", action="store_true")
parser.add_argument("--max_epoch", default=10, type=int)
ns = parser.parse_args()

torch.set_default_device(ns.device)
match ns.dataset:
    case "mnist":
        transform = transforms.Compose([transforms.ToTensor(),
            lambda x: (x > 0.5).float(), train.DevicePlacement()])
        dataset = datasets.MNIST(ns.datasets_folder, train=True, download=False, transform=transform)
        data_split = [55000, 5000]
        event_shape = [1, 28, 28]
        channel_layer = bernoulli_layer.IndependentBernoulli(event_shape=event_shape[:1])
    case "cifar10":
        num_buckets = 8
        transform = transforms.Compose([transforms.ToTensor(),
            transforms.Lambda(
                lambda value: qd.discretize(value, num_buckets)),
                train.DevicePlacement()])
        dataset = datasets.CIFAR10(ns.datasets_folder, train=True, download=False, transform=transform)
        data_split = [45000, 5000]
        event_shape = [3, 32, 32]
        channel_layer = ql.IndependentQuantizedDistribution(event_shape=event_shape[:1], add_noise=False)
    case _:
        raise RuntimeError("dataset {ns.dataset} not recognized.")

match ns.net:
    case "simple_pixelcnn_net": net = simple_pixelcnn_net.SimplePixelCNNNet(event_shape[0], channel_layer.params_size(), None)
    case "pixelcnn_net":  net = pixelcnn_net_pp.PixelCNN(nr_resnet=ns.num_resnet, nr_filters=160,
            input_channels=event_shape[0], nr_params=channel_layer.params_size(), nr_conditional=None)
    case _: raise RuntimeError("{ns.net} net name not recognized.")
image_distribution = pixelcnn.PixelCNN(net, event_shape, channel_layer, None)

train_dataset, validation_dataset = random_split(dataset, data_split,
    generator=torch.Generator(device=torch.get_default_device()))
tb_writer = SummaryWriter(ns.tb_folder)
epoch_end_callbacks = [
    callbacks.log_image_cb(pygen_models_callbacks.sample_images(image_distribution),
                           tb_writer=tb_writer, folder=ns.images_folder, name="generated_images"),
    callbacks.tb_epoch_log_metrics(tb_writer),
    callbacks.tb_dataset_metrics_logging(tb_writer, "validation", validation_dataset)]
train.train(image_distribution, train_dataset, pygen_models_train.distribution_objective,
    batch_end_callback=callbacks.tb_batch_log_metrics(tb_writer),
    epoch_end_callback=callbacks.callback_compose(epoch_end_callbacks), dummy_run=ns.dummy_run, max_epoch=ns.max_epoch)
