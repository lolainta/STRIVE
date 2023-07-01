import argparse
import pickle
from generation.Dataset import ColDataset
import os


def parse_cfg():
    parser = argparse.ArgumentParser(
        prog="python3 src/filter_collision_data.py",
        description="Filter generated dataset from given pickle file",
    )
    parser.add_argument(
        "-d",
        "--dir",
        required=True,
        default=None,
        help="Dataset folder",
    )
    parser.add_argument(
        "-o",
        "--out",
        required=True,
        default=None,
        help="Output folder",
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


def filter_data(dataset: ColDataset) -> bool:
    if not dataset.filter_by_vel_acc():
        return False
    if not dataset.filter_by_collision():
        return False
    return True


def filter_dir(scene_dir: str, args):
    print(f"Start: {scene_dir}")
    cur = 0
    for file in os.listdir(scene_dir):
        with open(os.path.join(scene_dir, file), "rb") as f:
            dataset: ColDataset = pickle.load(f)
        if filter_data(dataset):
            out_dir = os.path.join(
                args.out, args.version, dataset.type.name, dataset.scene["name"]
            )
            out = os.path.join(out_dir, f"{cur:02d}.pickle")
            os.makedirs(out_dir, exist_ok=True)
            with open(out, "wb") as f:
                pickle.dump(dataset, f)
            cur += 1
    print(f"Finished: {scene_dir} {len(os.listdir(scene_dir))} -> {cur}")


def main():
    args = parse_cfg()
    target_dir = os.path.join(args.out, args.version)
    os.makedirs(target_dir, exist_ok=True)
    for root, dir, files in os.walk(args.dir):
        files.sort()
        if len(files) and all(f[-7:] == (".pickle") for f in files):
            filter_dir(root, args)


if __name__ == "__main__":
    main()
