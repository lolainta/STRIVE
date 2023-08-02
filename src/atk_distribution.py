import argparse
import pickle
from generation.Dataset import ColDataset
import os
import collections
import tqdm
from generation.Drawer import Drawer
from nuscenes.nuscenes import NuScenes
from generation.NuscData import NuscData
from tqdm import trange
from nuscenes.map_expansion.map_api import NuScenesMap
from generation.Condition import Condition


def parse_cfg():
    parser = argparse.ArgumentParser(
        prog="python3 src/atk_distribution.py",
        description="Analyse the distribution of attacker",
    )
    parser.add_argument(
        "-d",
        "--dir",
        required=True,
        default=None,
        help="Dataset folder",
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
    args = parse_cfg()
    data_dir = os.path.join(args.dir, args.version)
    pickles = list()
    for root, dir, files in os.walk(data_dir):
        for file in files:
            if file[-7:] == ".pickle":
                pickles.append(os.path.join(root, file))
    print(f"Total: {len(pickles)}")
    # res = collections.defaultdict(int)
    dataCluster = list()
    for path in tqdm.tqdm(pickles):
        with open(path, "rb") as f:
            dataset: ColDataset = pickle.load(f)
            dataCluster.append(dataset)
            # dur = dataset.ego.datalist[-1].timestamp - dataset.ego.datalist[0].timestamp
            # dur = dur // 100000 / 10
            # res[dur] += 1

    dsdict = collections.defaultdict(list)
    for dataset in dataCluster:
        dsdict[dataset.scene["name"]].append(dataset)

    nusc_obj = NuScenes(
        version=f"v1.0-{args.version}",
        dataroot=f"data/nuscenes/{args.version}",
        verbose=True,
    )

    scene2data = dict()
    for i in trange(len(nusc_obj.scene)):
        nusc = NuscData(nusc_obj, i)
        mapName = nusc.get_map()
        nuscMap = NuScenesMap(
            dataroot=f"data/nuscenes/{args.version}",
            map_name=mapName,
        )
        scene2data[nusc_obj.scene[i]["name"]] = (nusc, nuscMap)

    colDict = {
        Condition.HO: "red",
        Condition.RE: "blue",
        Condition.LC: "green",
        Condition.JC: "yellow",
        Condition.LTAP: "purple",
    }
    for scene, datasets in dsdict.items():
        print(scene, len(datasets))
        drawer = Drawer(*scene2data[scene])
        atks = list()
        for dataset in datasets:
            atks.append((dataset.atk, colDict[dataset.cond]))
        out = os.path.join(data_dir, "analyse", "atks", f"{scene}.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        drawer.plot_atks(atks, out)


if __name__ == "__main__":
    main()
