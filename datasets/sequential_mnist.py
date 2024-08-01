import torch
from torch.utils.data import Dataset


class SequentialMNISTDataset(Dataset):
    """An element is a tensor of shape [4,1,28,28] representing a sequence of digit images of increasing value
    starting from a random digit and 9 wraps to 0.
    dummy_run uses small subset of MNIST, used just for checking program runs."""
    def __init__(self, mnist_dataset, seq_len=3, dummy_run=False):
        if dummy_run:
            self.mnist_dataset_len = 100
        else:
            self.mnist_dataset_len = len(mnist_dataset)
        mnist_digits = torch.stack([mnist_dataset[n][0] for n in range(self.mnist_dataset_len)]).float()
        mnist_labels = torch.tensor([mnist_dataset[n][1] for n in range(self.mnist_dataset_len)])
        self.digits = [mnist_digits[mnist_labels == d] for d in range(10)]
        self.seq_len = seq_len

    def __len__(self):
        return self.mnist_dataset_len

    def __getitem__(self, idx):
        d = torch.randint(low=0, high=9, size=[])
        rand_idx_sel = torch.randint(low=0, high=self.digits[d].shape[0], size=[])
        i1 = [self.digits[d][rand_idx_sel]]
        for s in range(self.seq_len-1):
            d = (d + 1) % 10
            rand_idx_sel = torch.randint(low=0, high=self.digits[d].shape[0], size=[])
            i1.append(self.digits[d][rand_idx_sel])
        return (torch.stack(i1),)
