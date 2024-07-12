# pygen_models

Contains some example models. Uses the training loop from pygen.

To train an mnist PixelCNN:

```
digit_distribution = pixelcnn.PixelCNNBernoulliDistribution(event_shape=[1, 28, 28])
train.DistributionTrainer(digit_distribution.to(ns.device), train_dataset).train()
```

./examples/mnist_pixelcnn.py: Generative MNIST model based on PixelCNN

./examples/cifar10_pixelcnn.py: Generative CIFAR10 model based on PixelCNN

The PixelCNN examples use the following repo:
```
git clone https://github.com/jfrancis71/pixel-cnn-pp.git
mv pixel-cnn-pp pixelcnn_pp
```
This is a modified fork of https://github.com/pclucas14/pixel-cnn-pp.
The modifications were made to support different types of probability distributions and to allow conditioning the distribution on a tensor.

## Installation and Setup

Install PyTorch (https://pytorch.org/get-started/locally/)

There is no package install currently setup, so you need to set PYTHONPATH to point to root of the repository, eg:

```
git clone https://github.com/jfrancis71/pygen.git
git clone https://github.com/jfrancis71/pygen_models.git
export PYTHONPATH=~/github_repos
```
