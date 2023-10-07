import glob
import logging
from pathlib import Path
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

logging.getLogger('PIL').setLevel(logging.WARNING)


def _tokenize(text):
    return text.split()


class SignDataset(Dataset):

    def __init__(
        self,
        data_root,
        data_split,
        transform=None,
        tokenize=_tokenize,
        lower=None,
        apply_background=False,
        eval_bg_dataset=None,
        eval_partition=None,
        error_check=False
    ):
        # sanity checks
        assert data_split in ["train", "dev", "test"]
        if apply_background:
            assert eval_bg_dataset is not None
            if data_split == "test":
                assert eval_partition is not None and eval_partition in [1, 2, 3]

        self.data_root = Path(data_root)
        self.data_split = data_split
        self.transform = transform
        self.tokenize = tokenize
        self.lower = lower
        self.apply_background = apply_background
        self.eval_bg_dataset = eval_bg_dataset
        self.eval_partition = eval_partition
        self.error_check = error_check

        self.dataset_dir = self.data_root / f"features/fullFrame-210x260px/{data_split}"
        self.ann_file = self.data_root / f"annotations/manual/{data_split}.corpus.csv"
        self.dataframe = pd.read_csv(self.ann_file, sep="|")
        self.glosses_list = self.tokenization(self.dataframe.annotation)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        video = self.dataset_dir / self.dataframe.iloc[index].folder

        if self.apply_background:
            video = str(video)[:-7] + self.data_split + '_' + self.eval_bg_dataset
            if self.data_split == "test":
                video = video + '_' + str(self.eval_partition) + '/*.png'
            elif self.data_split == "dev":
                video = video + '_1/*.png'
            else:
                raise ValueError

        if self.transform:
            video = self.transform(video)

        tokens = self.glosses_list[index]

        if self.error_check:
            errors = ["cl-", "loc-", "poss-", "qu-"]
            for i, t in enumerate(tokens):
                for e in errors:
                    if e in t:
                        tokens[i] = t.replace(e, "")

        indices = [self.vocab.stoi[token] for token in tokens]
        return video, indices, index

    def tokenization(self, annotation):
        glosses_list = []
        for i in range(len(annotation)):
            glosses_str = annotation[i]
            if self.tokenize is not None and isinstance(glosses_str, str):
                if self.lower:
                    glosses_str = glosses_str.lower()
                glosses = self.tokenize(glosses_str.rstrip("\n"))

            glosses_list.append(glosses)

        return glosses_list

    def video_length(self, idx):
        video = self.dataset_dir / self.dataframe.iloc[idx].folder
        return len(glob.glob(str(video)))

    def load_vocab(self, vocabulary):
        self.vocab = vocabulary
        self.pad_idx = self.vocab.stoi[self.vocab.pad_token]
        self.sil_idx = self.vocab.stoi[self.vocab.sil_token]

    def collate(self, data):
        videos, glosses, video_idx = list(zip(*data))

        def pad(videos, glosses):
            video_lengths = [len(v) for v in videos]  # maybe temporal dim
            max_video_len = max(video_lengths)
            extra_len = max_video_len % 4  # considering temporal pooling
            if extra_len != 0:
                max_video_len += 4 - extra_len

            padded_videos = []

            for video, length in zip(videos, video_lengths):
                C, H, W = video.size(1), video.size(2), video.size(3)
                new_tensor = video.new(max_video_len, C, H, W).fill_(0)
                new_tensor[:length] = video
                padded_videos.append(new_tensor)

            gloss_lengths = [len(s) for s in glosses]
            max_len = max(gloss_lengths)
            glosses = [
                s + [self.pad_idx] * (max_len - len(s)) if len(s) < max_len else s for s in glosses
            ]
            return (padded_videos, video_lengths, glosses, gloss_lengths)

        (videos, video_lengths, glosses, gloss_lengths) = pad(videos, glosses)
        videos = torch.stack(videos, dim=0)

        video_lengths = Tensor(video_lengths).long()
        glosses = Tensor(glosses).long()
        gloss_lengths = Tensor(gloss_lengths).long()

        return (videos, video_lengths, glosses, gloss_lengths, video_idx)
