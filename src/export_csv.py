from generation.Dataset import ColDataset
from generation.Condition import Condition
from generation.Data import Data
from argparse import ArgumentParser
import os
import pickle
import csv
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


def write_data_row(writer, data: Data, scene: str, identity: str):
    writer.writerow(
        [
            data.timestamp,
            identity,
            data.transform.translation.x,
            data.transform.translation.y,
            data.velocity,
            data.transform.rotation.yaw,
        ]
    )


def export(d: ColDataset, path: str):
    d.ego.grad()
    with open(path, "w", newline="\n") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(["TIMESTAMP", "TRACK_ID", "X", "Y", "V", "YAW"])
        for data in d.ego.datalist:
            write_data_row(writer, data, d.scene["name"], "ego")
        for data in d.atk.datalist:
            write_data_row(writer, data, d.scene["name"], d.inst["token"])
        for npc_tk, npc in zip(d.npc_tks, d.npcs):
            npc.grad()
            for data in npc.datalist:
                write_data_row(writer, data, d.scene["name"], npc_tk)


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
        export(
            dataset,
            os.path.join(target, f"{path.replace('/','_')}.csv"),
        )


if __name__ == "__main__":
    main()
