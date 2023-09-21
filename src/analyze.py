import argparse
import pickle
from generation.Dataset import ColDataset
import os
import collections
import tqdm
from generation.Drawer import Drawer
from nuscenes.nuscenes import NuScenes
from generation.NuscData import NuscData
from tqdm import trange, tqdm
from nuscenes.map_expansion.map_api import NuScenesMap
from generation.Condition import Condition
import json


def parse_cfg():
    parser = argparse.ArgumentParser(
        prog="python3 src/analyze.py",
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
        default="trainval",
        choices=["mini", "trainval"],
        help="Dataset version, mini or trainval",
    )
    args = parser.parse_args()
    print(args)
    return args


def main():
    args = parse_cfg()
    data_dir = os.path.join(args.dir, args.version)

    dataCluster = list()
    pickles = list()
    for root, dir, files in os.walk(data_dir):
        for file in files:
            if file[-7:] == ".pickle":
                pickles.append(os.path.join(root, file))
    for path in tqdm(pickles, desc="Loading", miniters=20, mininterval=0.5):
        with open(path, "rb") as f:
            dataset: ColDataset = pickle.load(f)
            dataCluster.append(dataset)
    print(f"Total: {len(dataCluster)}")
    res = collections.defaultdict(int)
    for dataset in dataCluster:
        dur = dataset.ego.datalist[-1].timestamp - dataset.ego.datalist[0].timestamp
        dur = dur // 500000 / 2
        res[dur] += 1
    print(json.dumps({k: res[k] for k in sorted(res)}))

    dsdict = collections.defaultdict(list)
    for dataset in dataCluster:
        dsdict[dataset.scene["name"]].append(dataset)

    nusc_obj = NuScenes(
        version=f"v1.0-{args.version}",
        dataroot=f"data/nuscenes/{args.version}",
        verbose=True,
    )

    scene2map = dict()
    maps = dict()
    for i in trange(len(nusc_obj.scene)):
        nusc = NuscData(nusc_obj, i)
        mapName = nusc.get_map()
        if mapName not in maps:
            nuscMap = NuScenesMap(
                dataroot=f"data/nuscenes/{args.version}",
                map_name=mapName,
            )
            maps[mapName] = nuscMap
        else:
            nuscMap = maps[mapName]
        scene2map[nusc_obj.scene[i]["name"]] = mapName

    colDict = {
        Condition.HO: "red",
        Condition.RE: "blue",
        Condition.LC: "green",
        Condition.JC: "orange",
        Condition.LTAP: "purple",
    }

    all_atks = collections.defaultdict(list)
    for scene, datasets in tqdm(dsdict.items()):
        tqdm.write(f"{scene}: {len(datasets)}")
        mapName = scene2map[scene]
        drawer = Drawer(maps[mapName])
        atks = list()
        for dataset in datasets:
            atks.append((dataset.atk, colDict[dataset.cond]))

        all_atks[mapName].extend(atks)

        out = os.path.join(data_dir, "analyse", "box", "scenes", f"{scene}.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        drawer.plot_atks(atks, out, no_box=False)

        out = os.path.join(data_dir, "analyse", "nobox", "scenes", f"{scene}.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        drawer.plot_atks(atks, out, no_box=True)

    for mapName, atks in all_atks.items():
        print(mapName, len(atks))
        drawer = Drawer(maps[mapName])

        out = os.path.join(data_dir, "analyse", "box", "maps", f"{mapName}.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        drawer.plot_atks(atks, out, no_box=False)


if __name__ == "__main__":
    main()
