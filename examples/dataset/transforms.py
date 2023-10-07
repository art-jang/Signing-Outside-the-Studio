import collections
import glob
import numpy as np
import random
import torch
import torchvision.transforms.functional as TF
from torchvision.io import read_image


class VideoFolderPathToTensor(object):

    def __init__(self):
        pass

    def __call__(self, path):
        frames_path = sorted(glob.glob(str(path)))
        frames = torch.stack([read_image(frames_path[i]) for i in range(len(frames_path))], dim=0)
        frames = TF.convert_image_dtype(frames, torch.float)

        return frames


class VideoResize(object):

    def __init__(self, size, interp="trilinear"):
        assert isinstance(size, collections.Iterable) and len(size) == 2
        self.size = size
        self.interp = interp

    def __call__(self, video):
        t, _, h, w = video.shape

        if self.interp in ["linear", "bilinear", "trilinear", "bicubic"]:
            align_corners = False
        else:
            align_corners = None

        new_size = (t, self.size[0], self.size[1])

        rescaled_video = torch.nn.functional.interpolate(
            video.transpose(0, 1).unsqueeze(dim=0),
            size=new_size,
            mode=self.interp,
            align_corners=align_corners,
        ).squeeze().transpose(0, 1)

        return rescaled_video

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class VideoRandomCrop(object):

    def __init__(self, size):
        assert len(size) == 2
        self.size = size

    def __call__(self, video):

        H, W = video.size()[2:]
        h, w = self.size
        assert H >= h and W >= w

        top = np.random.randint(0, H - h)
        left = np.random.randint(0, W - w)

        video = video[:, :, top:top + h, left:left + w]

        return video


class VideoCenterCrop(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, video):

        H, W = video.size()[2:]
        h, w = self.size
        assert H >= h and W >= w

        top = int((H - h) / 2)
        left = int((W - w) / 2)

        video = video[:, :, top:top + h, left:left + w]

        return video


class VideoRandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, video):

        if random.random() < self.p:
            video = video.flip([3])

        return video


class VideoRandomVerticalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, video):

        if random.random() < self.p:
            video = video.flip([2])

        return video


class VideoFrameAugmentation(object):

    def __init__(self, p_del=0.8, p_ins=0.8, max_ins_ratio=0.1, max_del_ratio=0.2):
        assert max_ins_ratio < 1.0 and max_del_ratio < 1.0
        self.max_ins_ratio = max_ins_ratio
        self.max_del_ratio = max_del_ratio
        self.p_ins = p_ins
        self.p_del = p_del

    def __call__(self, video):

        L, C, H, W = video.size()
        frames_inds = np.array([i for i in range(L)]).astype(np.int16)
        if random.random() < self.p_ins:
            ins_ratio = random.uniform(0.0, self.max_ins_ratio)
            rand_inds = np.random.choice(L, int(L * ins_ratio), replace=False)
            frames_inds = np.sort(np.concatenate([frames_inds, rand_inds], 0))

        if random.random() < self.p_del:
            del_ratio = random.uniform(0.0, self.max_del_ratio)
            rand_inds = np.random.choice(L, int(L * del_ratio), replace=False)
            frames_inds = np.delete(frames_inds, rand_inds)
        frames = torch.stack([video[i, :, :, :] for i in frames_inds], dim=0)

        return frames


class VideoNormalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, video):
        dtype = video.dtype
        device = video.device
        mean = torch.as_tensor(self.mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.std, dtype=dtype, device=device)
        if (std == 0).any():
            raise ValueError(
                'std evaluated to zero after conversion to {}, leading to division by zero.'.
                format(dtype)
            )
        if mean.ndim == 1:
            mean = mean[:, None, None]
        if std.ndim == 1:
            std = std[:, None, None]
        if video.ndim == 4:
            mean = mean[None, :, ...]
            std = std[None, :, ...]

        video.sub_(mean).div_(std)

        return video


class VideoBackground(object):

    def __init__(self, background_dataset, same=True):
        self.background_dataset = background_dataset
        self.same = same

    def __call__(self, clips):

        if self.same:
            background_list = []
            for _ in range(clips.size(0)):
                rand_idx = random.randint(0, len(self.background_dataset) - 1)
                background_img, _ = self.background_dataset[rand_idx]
                background_list.append(background_img.repeat([clips.size(1), 1, 1, 1]))

        backgrounds = torch.stack(background_list, dim=0).cuda()
        proportion = random.uniform(0.6, 0.95)
        clips = clips * proportion + backgrounds * (1 - proportion)

        return clips
