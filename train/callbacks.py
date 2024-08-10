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


class TBSequenceImageCallback:
    # pylint: disable=R0903
    def __init__(self, tb_writer, tb_name):
        self.tb_writer = tb_writer
        self.tb_name = tb_name

    def __call__(self, trainer_state):
        sample_size = 8
        imglist = [trainer_state.trainable.sample() for _ in range(sample_size)]
        imglist = torch.clip(torch.cat(imglist, axis=0), 0.0, 1.0)  # pylint: disable=E1101
        grid_image = torchvision.utils.make_grid(imglist, padding=10, nrow=trainer_state.trainable.num_steps)
        self.tb_writer.add_image(self.tb_name, grid_image, trainer_state.epoch_num)


class TBSequenceTransitionMatrixCallback:
    # pylint: disable=R0903
    def __init__(self, tb_writer, tb_name):
        self.tb_writer = tb_writer
        self.tb_name = tb_name

    def __call__(self, trainer_state):
        image = trainer_state.trainable.state_transition_distribution().probs.detach().unsqueeze(0).cpu().numpy()
        self.tb_writer.add_image(self.tb_name, image, trainer_state.epoch_num)


import doctest
doctest.testmod()
