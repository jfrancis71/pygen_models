import torch
import torchvision
from torchvision.utils import make_grid


def tb_sample_images(tb_writer, tb_name):
    """Creates a 4x4 grid of images by sampling the trainable.

    >>> callback = tb_sample_images(None, "")
    >>> base_distribution = torch.distributions.bernoulli.Bernoulli(logits=torch.zeros([1, 8, 8]))
    >>> distribution = torch.distributions.independent.Independent(base_distribution, reinterpreted_batch_ndims=3)
    >>> trainer = type('TrainingLoopInfo', (object,), {'trainable': distribution})()
    >>> callback(trainer)
    """
    def _fn(training_loop_info):
        imglist = training_loop_info.trainable.sample([16])
        grid_image = make_grid(imglist, padding=10, nrow=4, value_range=(0.0, 1.0))
        if tb_writer is not None:
            tb_writer.add_image(tb_name, grid_image, training_loop_info.epoch_num)
    return _fn


class TBSequenceImageCallback:
    # pylint: disable=R0903
    def __init__(self, tb_writer, tb_name):
        self.tb_writer = tb_writer
        self.tb_name = tb_name

    def __call__(self, trainer):
        sample_size = 8
        num_steps = 3
        imglist = [trainer.trainable.sample(num_steps=num_steps) for _ in range(sample_size)]
        imglist = torch.clip(torch.cat(imglist, axis=0), 0.0, 1.0)  # pylint: disable=E1101
        grid_image = torchvision.utils.make_grid(imglist, padding=10, nrow=num_steps)
        self.tb_writer.add_image(self.tb_name, grid_image, trainer.epoch)


class TBSequenceTransitionMatrixCallback:
    # pylint: disable=R0903
    def __init__(self, tb_writer, tb_name):
        self.tb_writer = tb_writer
        self.tb_name = tb_name

    def __call__(self, trainer):
        image = trainer.trainable.state_transition_distribution().probs.detach().unsqueeze(0).cpu().numpy()
        self.tb_writer.add_image(self.tb_name, image, trainer.epoch)
