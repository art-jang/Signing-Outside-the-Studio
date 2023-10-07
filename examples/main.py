import argparse

from dataset import buildDataloader


def arg_parser():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--data-root",
        default="data/phoenix2014-release/phoenix-2014-multisigner",
        type=str,
        help="path to PHOENIX-2014 dataset"
    )
    parser.add_argument("--train-batch-size", default=2, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    return parser


def main(args):
    data_loaders = buildDataloader(args)

    for data_split, data_loader in data_loaders.items():
        print(f"split: {data_split}, len: {len(data_loader.dataset)}")

    return


if __name__ == "__main__":
    args = arg_parser().parse_args()
    main(args)
