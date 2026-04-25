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
import pygen.layers.independent_bernoulli as bernoulli_layer
import pygen_models.layers.independent_quantized_distribution as ql
import pygen_models.distributions.quantized_distribution as qd
import pixelcnn_pp.model as pixelcnn_model
import pygen_models.layers.pixelcnn as pixelcnn
from pygen_models.neural_nets import simple_pixelcnn_net


def demo_classify_images(cond_image_layer, images, categories):
    """demos classifying images with generative model.

       Note: in implementation here I implicitly assume uniform prior distribution.
    """
    def _fn():
        log_probs = torch.stack([cond_image_layer(torch.arange(10)).log_prob(images[image_idx].unsqueeze(0). \
            repeat([10,1,1,1])).detach() for image_idx in range(len(images))])
        label_indices = torch.argmax(log_probs, dim=1)
        labels = [categories[idx.to("cpu").item()] for idx in label_indices]
        labelled_images = callbacks.make_labelled_images_grid(images, labels)
        return labelled_images
    return _fn


parser = argparse.ArgumentParser(description='PyGen Conditional PixelCNN')
parser.add_argument("--datasets_folder", default="~/datasets")
parser.add_argument("--dataset", default="mnist")
parser.add_argument("--tb_folder", default=None)
parser.add_argument("--images_folder", default=None)
parser.add_argument("--device", default="cpu")
parser.add_argument("--net", default="simple_pixelcnn_net")
parser.add_argument("--num_resnet", default=3, type=int)
parser.add_argument("--max_epoch", default=10, type=int)
parser.add_argument("--dummy_run", action="store_true")
parser.add_argument("--classifier", action="store_true")
ns = parser.parse_args()

num_pixelcnn_params = 16
torch.set_default_device(ns.device)

if ns.dataset == "mnist":
    transform = transforms.Compose([transforms.ToTensor(),
        lambda x: (x > 0.5).float(), train.DevicePlacement()])
    dataset = datasets.MNIST(ns.datasets_folder, train=True, download=False, transform=transform)
    event_shape = [1, 28, 28]
    data_split = [55000, 5000]
    channel_layer = bernoulli_layer.IndependentBernoulli(event_shape=event_shape[:1])
elif ns.dataset == "cifar10":
    num_buckets = 8
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda value: qd.discretize(value, num_buckets)), train.DevicePlacement()])
    dataset = datasets.CIFAR10(ns.datasets_folder, train=True, download=False, transform=transform)
    event_shape = [3, 32, 32]
    data_split = [45000, 5000]
    channel_layer = ql.IndependentQuantizedDistribution(event_shape=event_shape[:1], add_noise=False)
else:
    raise RuntimeError(f"{ns.dataset} not recognized.")

match ns.net:
    case "simple_pixelcnn_net": net = simple_pixelcnn_net.SimplePixelCNNNet(event_shape[0], channel_layer.params_size(), num_pixelcnn_params)
    case "pixelcnn_net":  net = pixelcnn_model.PixelCNN(nr_resnet=ns.num_resnet, nr_filters=160,
            input_channels=event_shape[0], nr_params=channel_layer.params_size(), nr_conditional=num_pixelcnn_params)
    case _: raise RuntimeError("{ns.net} net name not recognized.")

pixelcnn_layer = pixelcnn.PixelCNN(net, event_shape, channel_layer, num_pixelcnn_params)

train_dataset, validation_dataset = random_split(dataset, data_split,
    generator=torch.Generator(device=torch.get_default_device()))
example_valid_images = next(iter(torch.utils.data.DataLoader(validation_dataset, batch_size=25)))[0]
tb_writer = SummaryWriter(ns.tb_folder)
categorical_image_layer = nn.Sequential(
    one_hot.OneHot(10),
    pixelcnn.SpatialExpand(10, num_pixelcnn_params, event_shape[1:]),
    pixelcnn_layer)
epoch_end_callbacks = [
    callbacks.tb_epoch_log_metrics(tb_writer),
    callbacks.tb_dataset_metrics_logging(tb_writer, "validation", validation_dataset),
    callbacks.log_image_cb(callbacks.demo_conditional_images(categorical_image_layer, torch.arange(10), num_samples=2),
            tb_writer=tb_writer, folder=ns.images_folder, name="conditional_generated_images")
]
if ns.classifier:
    epoch_end_callbacks.append(callbacks.log_image_cb(demo_classify_images(categorical_image_layer, example_valid_images, dataset.classes),
        tb_writer=tb_writer, folder=ns.images_folder, name="classification"))
train.train(
    categorical_image_layer, train_dataset, train.layer_objective(reverse_inputs=True),
    batch_end_callback=callbacks.tb_batch_log_metrics(tb_writer),
    epoch_end_callback=callbacks.callback_compose(epoch_end_callbacks), dummy_run=ns.dummy_run, max_epoch=ns.max_epoch)
