from generation.Dataset import ColDataset
from argparse import ArgumentParser
import os
import pickle
import tqdm


def parse_cfg():
    parser = ArgumentParser(
        prog="python3 src/export_csv.py",
        description="Export csv files from pickles",
    )
    parser.add_argument(
        "-d",
        "--dir",
        required=True,
        default=None,
        help="Collision Data Folder",
    )
    parser.add_argument(
        "-v",
        "--version",
        required=True,
        default="mini",
        choices=["mini", "trainval"],
        help="Dataset version, mini or trainval",
    )
    args = parser.parse_args()
    print(args)
    return args


def main():
    cfg = parse_cfg()

    data_dir = os.path.join(cfg.dir, cfg.version)
    pickles = list()
    for root, dir, files in os.walk(data_dir):
        for file in files:
            if file[-7:] == ".pickle":
                pickles.append(os.path.join(root, file))
    print(f"Total: {len(pickles)}")

    target = os.path.join(cfg.dir, cfg.version, "csvs")
    os.makedirs(target, exist_ok=True)
    for path in tqdm.tqdm(pickles):
        with open(path, "rb") as f:
            dataset: ColDataset = pickle.load(f)
        assert file[-7:] == ".pickle"
        dataset.export(
            os.path.join(target, f"{path.replace('/','_')[:-7]}.csv"),
        )


if __name__ == "__main__":
    main()
