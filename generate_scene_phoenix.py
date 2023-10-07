import argparse
import glob
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Subset

import pandas as pd
from bg_datasets import LSUN, SUN397
from models import UNet
from utils import utils


def build_segmentation_model(pretrained: str):
    model = UNet(backbone="mobilenetv2", num_classes=2, pretrained_backbone=None)
    model = model.cuda()
    trained_dict = torch.load(pretrained, map_location="cpu")['state_dict']
    model.load_state_dict(trained_dict, strict=False)
    model.eval()
    return model


def load_phoenix_dataset(data_root: str, sign_split: str = "test"):
    """
        sign_split (str): "dev" or "test" of phoenix dataset
    """
    img_prefix = os.path.join(data_root, "features/fullFrame-210x260px", sign_split)
    ann_file = "annotations/manual/{}.corpus.csv".format(sign_split)
    annotations = pd.read_csv(os.path.join(data_root, ann_file), sep="|")

    return img_prefix, annotations


def build_background_test_set(bg_root, background_type, partition=1):
    assert background_type in ["LSUN", "SUN397"]

    # test transform
    test_transform = transforms.Compose([transforms.Resize(260), transforms.CenterCrop((210, 260))])

    if background_type == "LSUN":
        bg_dataset = LSUN(bg_root, classes="val", transform=test_transform)
    else:
        bg_dataset = SUN397(bg_root, split="Testing", partition=partition, transform=test_transform)
    return bg_dataset


def process_video_with_bg(model, sign_images_path, bg_img, save_dir):
    # runs phoenix test sample-level
    images_list = glob.glob(sign_images_path)
    images_list.sort()

    # parameters for image matting.
    KERNEL_SZ = 15
    SIGMA = 3.0

    def sem_seg_inference(model, image):
        # one frame inference for mask
        h, w = image.shape[:2]
        X, pad_up, pad_left, h_new, w_new = utils.preprocessing(
            image, expected_size=224, pad_value=0
        )
        with torch.no_grad():
            mask = model(X.cuda())
            mask = mask[..., pad_up:pad_up + h_new, pad_left:pad_left + w_new]
            mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=True)
            mask = F.softmax(mask, dim=1)
            mask = mask[0, 1, ...].detach().cpu().numpy()
        return mask

    # frame-level inference
    for i, img_path in enumerate(images_list):
        # one single sign frame from one video
        img = np.asarray(Image.open(img_path).convert("RGB"))
        mask = sem_seg_inference(model, img)

        synthesized_img = utils.draw_fore_to_back(
            img, mask, bg_img, kernel_sz=KERNEL_SZ, sigma=SIGMA
        )
        synthesized_img = Image.fromarray(synthesized_img)  # PIL

        save_dir = os.path.join("/".join(img_path.split('/')[:-2]), save_dir)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, img_path.split('/')[-1])

        synthesized_img.save(save_path)


def generate_sign_test_dataset_with_background(
    sign_root,
    bg_dataset,
    sign_split: str = "test",
    background_type: str = "LSUN",
    save_dir: str = "",
    partition: int = 1
):
    """
        sign_root (str): the root path for phoenix dataset
        bg_dataset (Dataset): the background (bg) data used for synthesizing sign samples 
            with bg. (LSUN or SUN397)
        sign_split (str): "dev" or "test" for PHOENIX evaluation dataset
        background_type (str): the kinds of background dataset. "LSUN" or "SUN397" 
        save_dir (str): where to save the videos
        partition (int): for generating multiple splits  
    """

    # BUILD SEGMENTATION MODEL
    model = build_segmentation_model("./pretrained/UNet_MobileNetV2.pth")

    # BUILD PHOENIX DATASET
    sign_img_prefix, sign_anns = load_phoenix_dataset(sign_root, sign_split=sign_split)

    # LOAD PRE-DEFINED BACKGROUND IMAGES FOR TEST SAMPLES
    split_file = "phoenix_{}_background_{}_{}.txt".format(sign_split, background_type, partition)
    split_path = os.path.join(bg_dataset.root, split_file)
    assert os.path.exists(split_path
                          ), "Check pre-defined background data split: {}".format(split_path)

    print()
    print("loaded pre-defined background sample indices from: {}".format(split_path))
    print()
    with open(split_path, "r") as f:
        bg_indices = f.read().splitlines()
    bg_indices = [int(ind) for ind in bg_indices]

    # background subset whose length corresponds to the phoenix evaluation set
    assert len(sign_anns) == len(bg_indices)
    background_test = Subset(bg_dataset, bg_indices)

    for i in range(len(background_test)):
        print("{} / {}".format(i + 1, len(background_test)))
        sign_images_path = os.path.join(sign_img_prefix, sign_anns["folder"][i])
        bg_img, _ = background_test[i]
        bg_img = np.asarray(bg_img)

        process_video_with_bg(model, sign_images_path, bg_img, save_dir)


def main(args):
    # fix the randomness of segmentation network
    partition = args.partition
    # options for sign dataset
    sign_root = args.sign_root
    sign_split = args.sign_split
    background_type = args.background_type
    # options for bg dataset

    if background_type == "LSUN":
        bg_root = "data/lsun"  # path to LSUN dataset
    elif background_type == "SUN397":
        bg_root = "data/SUN397"  # path to SUN397 dataset
    else:
        raise NotImplementedError

    save_dir = sign_split + "_" + background_type + "_" + str(partition)

    bg_dataset = build_background_test_set(bg_root, background_type, partition=partition)

    generate_sign_test_dataset_with_background(
        sign_root,
        bg_dataset,
        sign_split=sign_split,
        background_type=background_type,
        save_dir=save_dir,
        partition=partition
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--sign_root",
        default="data/phoenix2014-release/phoenix-2014-multisigner",
        type=str,
        help="path to sign dataset video"
    )
    parser.add_argument("--sign_split", default="dev", type=str, help="split dataset (dev / test)")
    parser.add_argument(
        "--background_type",
        default="SUN397",
        type=str,
        help="background data type (LSUN / SUN397)"
    )
    parser.add_argument(
        "--partition", default=1, type=int, help="partition number for split (1, 2, or 3)"
    )
    args = parser.parse_args()
    main(args)
