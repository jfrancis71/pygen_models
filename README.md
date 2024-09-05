# pygen_models

Contains some example models. Uses the training loop from pygen.

Current Status: 05/09/2024. This is a personal project (you are welcome to use if find useful) and is in a very immature state of development. The pixelcnn and conditional pixelcnn examples are somewhat stable, but generally other examples are highly experimental and speculative and probably not of general interest at the moment.

To train an mnist PixelCNN:

```
digit_distribution = pixelcnn.make_pixelcnn(
            pixelcnn.make_bernoulli_base_distribution(), net, event_shape=[1, 28, 28])
train.train(digit_distribution, train_dataset, pygen_models_train.distribution_objective)
```

./examples/pixelcnn.py: Generative model based on PixelCNN (cmd option to train MNIST or CIFAR10)


## Installation and Setup

Install PyTorch (https://pytorch.org/get-started/locally/)

There is no package install currently setup, so you need to set PYTHONPATH to point to root of the repository, eg:

```
git clone https://github.com/jfrancis71/pygen.git
git clone https://github.com/jfrancis71/pygen_models.git
export PYTHONPATH=~/github_repos
```

The PixelCNN examples use the following repo:
```
git clone https://github.com/jfrancis71/pixel-cnn-pp.git
mv pixel-cnn-pp pixelcnn_pp
```
This is a modified fork of https://github.com/pclucas14/pixel-cnn-pp.
The modifications were made to support different types of probability distributions and to allow conditioning the distribution on a tensor.
