from pygen.train import train


class RegTrainer(train.DistributionTrainer):
    def __init__(self, trainable, dataset, batch_size=32, max_epoch=10, batch_end_callback=None,
                 epoch_end_callback=None, use_scheduler=False, dummy_run=False, model_path=None):
        super().__init__(
            trainable, dataset, batch_size, max_epoch, batch_end_callback,
            epoch_end_callback, use_scheduler=use_scheduler, dummy_run=dummy_run,
            model_path=model_path)

    def batch_log_prob(self, batch):
        log_prob = self.trainable.log_prob(batch[0].to(self.device))
        reg = self.trainable.reg(batch[0].to(self.device))/(self.epoch+1)
        return log_prob -reg + reg.detach()


class VAETrainer(train.DistributionTrainer):
    def __init__(self, trainable, dataset, batch_size=32, max_epoch=10, batch_end_callback=None,
                 epoch_end_callback=None, use_scheduler=False, dummy_run=False, model_path=None):
        super().__init__(
            trainable, dataset, batch_size, max_epoch, batch_end_callback,
            epoch_end_callback, use_scheduler=use_scheduler, dummy_run=dummy_run,
            model_path=model_path)
        self.start_epoch()

    def start_epoch(self):
        self.total_reconstruct_log_prob = 0.0
        self.total_kl_div = 0.0
        super().start_epoch()

    def batch_log_prob(self, batch):
        log_prob, reconstruct_log_prob, kl_div = self.trainable.elbo(batch[0].to(self.device))
        self.reconstruct_log_prob_item = reconstruct_log_prob.mean().item()
        self.total_reconstruct_log_prob += self.reconstruct_log_prob_item
        self.kl_div_item = kl_div.mean().item()
        self.total_kl_div += self.kl_div_item
        reg = self.trainable.reg(batch[0].to(self.device))/(self.epoch+1)
        return log_prob - reg + reg.detach()
