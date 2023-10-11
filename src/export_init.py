import argparse
import os
from nuscenes.nuscenes import NuScenes
from generation.NuscData import NuscData
from tqdm import trange


def parse_cfg():
    parser = argparse.ArgumentParser(
        prog="python3 src/export_init.py",
        description="Export initial data from nuscenes data",
    )
    parser.add_argument(
        "-d",
        "--dir",
        required=True,
        default=None,
        help="Output folder",
    )
    parser.add_argument(
        "-v",
        "--version",
        default="trainval",
        choices=["mini", "trainval"],
        help="Dataset version, mini or trainval",
    )
    args = parser.parse_args()
    print(args)
    return args


def main():
    cfg = parse_cfg()
    print("Loading NuScenes dataset...")
    nusc = NuScenes(
        version=f"v1.0-{cfg.version}",
        dataroot=f"data/nuscenes/{cfg.version}",
        verbose=True,
    )
    data_dir = os.path.join(cfg.dir, f"init-{cfg.version}")
    os.makedirs(data_dir, exist_ok=True)
    print(f"Exporting to {data_dir}...")
    for i in trange(len(nusc.scene)):
        nuscData = NuscData(nusc, i)
        nuscData.export(os.path.join(data_dir, nuscData.scene["name"]))


if __name__ == "__main__":
    main()
