import sys
from collections import OrderedDict
from random import shuffle
import torchvision.transforms as TF
from torch.utils.data import DataLoader, Sampler

import transforms

from .datasets import SignDataset
from .vocabulary import build_vocab

NORMALIZE_MEAN = (0.5371, 0.5272, 0.5195)
NORMALIZE_STD = (0.2839, 0.2932, 0.3238)


def _buildDataset(data_root):
    norm_params = dict(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)

    datasets = {}

    # load original PHOENIX-2014 datasets
    tfm_train = TF.Compose(
        [
            transforms.VideoFolderPathToTensor(),
            transforms.VideoResize([256, 256]),
            transforms.VideoRandomCrop([224, 224]),
            transforms.VideoRandomHorizontalFlip(),
            transforms.VideoFrameAugmentation(),
            transforms.VideoNormalize(**norm_params)
        ]
    )
    datasets["train"] = SignDataset(data_root, "train", transform=tfm_train, apply_background=False)

    tfm_eval = TF.Compose(
        [
            transforms.VideoFolderPathToTensor(),
            transforms.VideoResize([256, 256]),
            transforms.VideoCenterCrop([224, 224]),
            transforms.VideoNormalize(**norm_params)
        ]
    )
    datasets["val"] = SignDataset(data_root, "dev", transform=tfm_eval, apply_background=False)
    datasets["test"] = SignDataset(data_root, "test", transform=tfm_eval, apply_background=False)

    # load evaluation datasets with synthesize backgrounds
    for bg_dataset in ["LSUN", "SUN397"]:
        # validation split
        datasets[f"val_bg_{bg_dataset}"] = SignDataset(
            data_root, "dev", eval_bg_dataset=bg_dataset, transform=tfm_eval, apply_background=True
        )

        # test splits
        test_partitions = [1, 2, 3]
        for partition in test_partitions:
            dataset = SignDataset(
                data_root,
                "test",
                eval_bg_dataset=bg_dataset,
                eval_partition=partition,
                transform=tfm_eval,
                apply_background=True
            )
            datasets[f"test_bg_{bg_dataset}_part{partition}"] = dataset

    # build vocabulary and apply to respective dataset
    vocab = build_vocab(datasets["train"], sys.maxsize, min_freq=1)
    for data_split in datasets:
        datasets[data_split].load_vocab(vocab)

    return datasets


class BucketBatchSampler(Sampler):

    def __init__(self, train_dataset, batch_size, drop_last=False):
        self.batch_size = batch_size
        self.drop_last = drop_last
        ind_n_len = []

        for idx in range(len(train_dataset)):
            ind_n_len.append((idx, train_dataset.video_length(idx)))

        self.ind_n_len = ind_n_len
        self.batch_list = self._generate_batch_map()
        self.num_batches = len(self.batch_list) if not self.drop_last else len(self.batch_list) - 1

    def _generate_batch_map(self):
        shuffle(self.ind_n_len)
        batch_map = OrderedDict()
        for idx, length in self.ind_n_len:
            if length not in batch_map:
                batch_map[length] = [idx]
            else:
                batch_map[length].append(idx)
        flattened_map = []
        for key in sorted(batch_map.keys()):
            flattened_map.extend(batch_map[key])

        batch_list = []
        for i in range(0, len(flattened_map), self.batch_size):
            batch_list.append(flattened_map[i:i + self.batch_size])
        return batch_list

    def batch_count(self):
        return self.num_batches

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        batch_list = self._generate_batch_map()
        shuffle(batch_list)
        for batch_inds in batch_list:
            if len(batch_inds) < self.batch_size and self.drop_last:
                continue
            yield batch_inds


def buildDataloader(args):
    data_root = args.data_root
    train_batch_size = args.train_batch_size
    num_workers = args.num_workers

    # build datasets
    datasets = _buildDataset(data_root)

    # build data_loaders
    data_loaders = {}
    for data_split, dataset in datasets.items():
        if "train" in data_split:
            data_loaders[data_split] = DataLoader(
                dataset=dataset,
                collate_fn=dataset.collate,
                batch_sampler=BucketBatchSampler(dataset, train_batch_size),
                shuffle=False,  # batch sampler includes random shuffling
                drop_last=False,
                num_workers=num_workers,
                pin_memory=True,
            )
        else:
            data_loaders[data_split] = DataLoader(
                dataset=dataset,
                collate_fn=dataset.collate,
                batch_size=1,
                shuffle=False,
                drop_last=False,
                num_workers=num_workers,
            )

    return data_loaders
