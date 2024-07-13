import torch
import torchvision


class TBDatasetVAECallback:
    def __init__(self, tb_writer, tb_name, dataset, batch_size=32):
        self.tb_writer = tb_writer
        self.tb_name = tb_name
        self.batch_size = batch_size
        self.dataset = dataset

    def __call__(self, trainer):
        dataloader = torch.utils.data.DataLoader(
            self.dataset, collate_fn=None,
            batch_size=self.batch_size, shuffle=True, drop_last=True)
        reconstruct_log_prob = 0.0
        kl = 0.0
        size = 0
        for (_, batch) in enumerate(dataloader):
            _, batch_reconstruct_log_prob, batch_kl = trainer.trainable.elbo(batch[0].to(trainer.device))
            reconstruct_log_prob += batch_reconstruct_log_prob.mean().item()
            kl += batch_kl.mean().item()
            size += 1
        self.tb_writer.add_scalar(self.tb_name+"_reconstruct", reconstruct_log_prob/size, trainer.epoch)
        self.tb_writer.add_scalar(self.tb_name + "_kl", kl / size, trainer.epoch)


class TBBatchVAECallback:
    def __init__(self, tb_writer):
        self.tb_writer = tb_writer

    def __call__(self, trainer):
        self.tb_writer.add_scalar("batch_reconstruct_log_prob", trainer.reconstruct_log_prob_item, trainer.epoch)
        self.tb_writer.add_scalar("batch_kl_div", trainer.kl_div_item, trainer.epoch)


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
