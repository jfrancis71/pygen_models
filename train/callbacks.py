import torch
import torchvision
from torchvision.utils import make_grid


def sample_images(distribution):
    """Creates a 4x4 grid of images by sampling the trainable.

    Example:
    >>> base_distribution = torch.distributions.bernoulli.Bernoulli(logits=torch.zeros([1, 8, 8]))
    >>> distribution = torch.distributions.independent.Independent(base_distribution, reinterpreted_batch_ndims=3)
    >>> sampling_fn = sample_images(distribution)
    >>> len(sampling_fn().shape)
    3
    """
    def _fn():
        img_list = distribution.sample([16])
        grid_image = make_grid(img_list, padding=10, nrow=4, value_range=(0.0, 1.0))
        return grid_image
    return _fn


import doctest
doctest.testmod()
