import argparse
import pickle
from generation.Drawer import Drawer
from generation.NuscData import NuscData
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
import glob, os
from multiprocessing import Pool


def show(path: str, nuscs, args):
    print(f"Loading {path}")
    with open(path, "rb") as f:
        dataset = pickle.load(f)
    found = False
    for nuscData, nuscMap in nuscs:
        if nuscData.scene["name"] == dataset.scene["name"]:
            plt = Drawer(nuscData, nuscMap)
            plt.plot_dataset(dataset, args.out, path)
            plt.close()
            found = True
            print(f"Saved: {path} {dataset.inst['token']}", flush=True)
    if not found:
        assert False, f"Scene {dataset.scene['name']} not found"


def main():
    parser = argparse.ArgumentParser(
        prog="python3 src/visualize.py",
        description="Visualize generated dataset from given pickle file",
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
    args = parser.parse_args()
    print(args)
    nusc_obj = NuScenes(
        version="v1.0-mini", dataroot="data/nuscenes/mini", verbose=True
    )
    nuscs = list()
    for i in range(len(nusc_obj.scene)):
        nuscData = NuscData(nusc_obj, i)
        mapName = nuscData.get_map()
        nuscMap = NuScenesMap(
            dataroot="data/nuscenes/mini",
            map_name=mapName,
        )
        nuscs.append((nuscData, nuscMap))
    pickles = glob.glob(f"./{args.dir}/**/*.pickle", recursive=True)
    params = list()
    for p in pickles:
        params.append((p, nuscs, args))
    with Pool(8) as pool:
        pool.starmap(show, params)


if __name__ == "__main__":
    main()
