# SUN397 datasets
import numpy as np
import os
from typing import Callable, List, Optional, Union
from PIL import Image
from torchvision.datasets.lsun import LSUN as _LSUN
from torchvision.datasets.vision import VisionDataset as _VisionDataset


# Base class for background dataset
class VisionDataset(_VisionDataset):

    def __init__(
        self,
        root,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        super(VisionDataset,
              self).__init__(root, transform=transform, target_transform=target_transform)

    def show_class_count(self):
        assert self.dbs is not None
        assert self.classes is not None

        counts = []
        for c, class_name in enumerate(self.classes):
            print("{}: {}".format(class_name, len(self.dbs[c])))
            counts.append(len(self.dbs[c]))

        min_ind, max_ind = np.argmin(counts), np.argmax(counts)
        print(
            "min: {} ({}),  max: {} ({})".format(
                counts[min_ind], self.classes[min_ind], counts[max_ind], self.classes[max_ind]
            )
        )

    def get_subset_indices(self, K: int, random_sample=False, shuffle=False):
        """
            Per-class subset index selection.
            This is used for both train and test time.
        """
        indices = []
        base_indices = list(range(K))

        count = 0
        for c in range(self.num_classes):
            assert K <= len(self.dbs[c])
            if random_sample:
                base_indices = np.random.permutation(len(self.dbs[c]))[:K]
            indices.extend([count + base_ind for base_ind in base_indices])
            count = self.indices[c]
        if shuffle:
            np.random.shuffle(indices)
        return indices


# LSUN dataset
class LSUN(_LSUN):

    def __init__(
        self,
        root: str,
        classes: Union[str, List[str]] = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super(LSUN, self).__init__(
            root, classes=classes, transform=transform, target_transform=target_transform
        )
        self.num_classes = len(self.classes)

    def show_class_count(self):
        assert self.dbs is not None
        assert self.classes is not None

        counts = []
        for c, class_name in enumerate(self.classes):
            print("{}: {}".format(class_name, len(self.dbs[c])))
            counts.append(len(self.dbs[c]))

        min_ind, max_ind = np.argmin(counts), np.argmax(counts)
        print(
            "min: {} ({}),  max: {} ({})".format(
                counts[min_ind], self.classes[min_ind], counts[max_ind], self.classes[max_ind]
            )
        )

    def get_subset_indices(self, K: int, random_sample=False, shuffle=False):
        """
            Per-class subset index selection
        """
        indices = []
        base_indices = list(range(K))

        count = 0
        for c in range(self.num_classes):
            assert K <= len(self.dbs[c])
            if random_sample:
                base_indices = np.random.permutation(len(self.dbs[c]))[:K]
            indices.extend([count + base_ind for base_ind in base_indices])
            count = self.indices[c]
        if shuffle:
            np.random.shuffle(indices)
        return indices


class SUN397Class(VisionDataset):

    def __init__(
        self,
        root,
        examples: List[str],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        super(SUN397Class,
              self).__init__(root, transform=transform, target_transform=target_transform)
        self.examples = examples

    def __getitem__(self, index):
        img, target = None, None
        img_path = (self.root + self.examples[index])  # as a special case
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.examples)


class SUN397(VisionDataset):
    """`SUN397 <https://vision.princeton.edu/projects/2010/SUN/>`_ dataset.

    Args:
        root (string): Root directory for the database files.
        split (str): either `Training` or `Testing` set
        partition (int): A partition index to be used for train and test set, between 1 and 10.
        classes (list): a list of categories to load, specified in ClassName.txt. 
            e,g. ['/a/abbey', '/a/airplane_cabin'].
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

    reference from https://pytorch.org/vision/stable/_modules/torchvision/datasets/lsun.html#LSUN
    """

    def __init__(
        self,
        root: str,
        split: str = "Training",
        partition: int = 1,
        classes: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super(SUN397, self).__init__(root, transform=transform, target_transform=target_transform)
        self.classes = self._verify_classes(classes)
        self.num_classes = len(self.classes)

        # for each class, create an LSUNClassDataset
        self.create_dbs(split, partition)

    def _verify_classes(self, classes: Optional[List[str]] = None):
        # read the whole class names
        class_names_file = os.path.join(self.root, "Partitions", "ClassName.txt")
        assert os.path.exists(class_names_file)
        with open(class_names_file, "r") as f:
            class_names = f.read().splitlines()
        assert len(class_names) == 397

        if classes is not None:
            assert isinstance(classes, list)
            for class_name in classes:
                assert class_name in class_names
            class_names = classes
        return class_names

    def create_dbs(self, split, partition):
        split_file = os.path.join(self.root, "Partitions", "{}_{:02d}.txt".format(split, partition))
        print(split_file)
        assert os.path.exists(split_file)

        # load partition
        with open(split_file, "r") as f:
            examples = f.read().splitlines()

        # database
        dbs = []
        for c, class_name in enumerate(self.classes):
            sub_examples = [_path for _path in examples if _path.startswith(class_name)]
            dbs.append(
                SUN397Class(
                    self.root,
                    sub_examples,
                    transform=self.transform,
                    target_transform=self.target_transform
                )
            )
        self.dbs = dbs

        # indices
        indices = []
        count = 0
        for db in self.dbs:
            count += len(db)
            indices.append(count)
        self.indices = indices
        self.length = count

    def __getitem__(self, index):
        target = 0
        sub = 0
        for ind in self.indices:  # each element is the end index of each db (class)
            if index < ind:
                break
            target += 1
            sub = ind  # how many db (in index) passed? -> net index: index - sub

        # select db
        db = self.dbs[target]
        index = index - sub

        if self.target_transform is not None:
            target = self.target_transform(target)

        img, _ = db[index]

        return img, target

    def __len__(self):
        return self.length
