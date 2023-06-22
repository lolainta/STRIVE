from datasets.collision_dataset import CollisionDataset
from generation.Dataset import ColDataset
from generation.Condition import Condition
from argparse import ArgumentParser
import os
import pickle
import csv


def parse_cfg():
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--data",
        default="data/hcis",
        help="Collision Data Path",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="data/hcis",
        help="Output Path",
    )
    args = parser.parse_args()
    return args


def export(d: ColDataset, path: str):
    d.ego.gen_velocity()
    with open(path, "w", newline="\n") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(["scene", "timestamp", "track", "x", "y", "v", "yaw"])
        for data in d.ego.datalist:
            writer.writerow(
                [
                    d.scene["name"],
                    data.timestamp,
                    "ego",
                    data.transform.translation.x,
                    data.transform.translation.y,
                    data.velocity,
                    data.transform.rotation.yaw,
                ]
            )
        for data in d.atk.datalist:
            writer.writerow(
                [
                    d.scene["name"],
                    data.timestamp,
                    d.inst["token"],
                    data.transform.translation.x,
                    data.transform.translation.y,
                    data.velocity,
                    data.transform.rotation.yaw,
                ]
            )
    pass


def main():
    cfg = parse_cfg()
    target = os.path.join(cfg.output, "csvs", "trainval")
    os.makedirs(target, exist_ok=True)
    for root, dir, files in os.walk(cfg.data):
        for file in files:
            scene = os.path.join(root, file)
            with open(scene, "rb") as f:
                dataset: ColDataset = pickle.load(f)
                export(
                    dataset,
                    os.path.join(
                        target, f"{root.replace('/','_')}_{dataset.inst['token']}"
                    ),
                )


if __name__ == "__main__":
    main()
