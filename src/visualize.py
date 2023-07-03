import argparse
import pickle
from generation.Drawer import Drawer
from generation.NuscData import NuscData
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
import glob, os
from multiprocessing import Process, Semaphore
from tqdm import trange
from time import sleep


def show(path: str, out_dir, nuscs):
    out = os.path.join(out_dir, path.replace("/", "_")[12:-7])
    if os.path.exists(f"{out}.mp4"):
        print(f"Skip: {path}", flush=True)
        return
    print(f"Loading: {path}", flush=True)
    with open(path, "rb") as f:
        dataset = pickle.load(f)
    f.close()
    found = False
    for nuscData, nuscMap in nuscs:
        if nuscData.scene["name"] == dataset.scene["name"]:
            found = True
            plt = Drawer(nuscData, nuscMap)
            plt.plot_dataset(dataset, out)
            plt.close()
            print(f"Saved: {path} {dataset.inst['token']}", flush=True)
    if not found:
        assert False, f"Scene {dataset.scene['name']} not found"


def sem_show(sem: Semaphore, path: str, out_dir, nuscs):
    sem.acquire()
    show(path, out_dir, nuscs)
    sem.release()


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
        "-v",
        "--version",
        required=True,
        default="mini",
        choices=["mini", "trainval"],
        help="Data version",
    )
    args = parser.parse_args()
    print(args)
    nusc_obj = NuScenes(
        version=f"v1.0-{args.version}",
        dataroot=f"data/nuscenes/{args.version}",
        verbose=True,
    )
    out_dir = os.path.join(args.dir, args.version, "viz")
    maps = dict()
    nuscs = list()
    for i in trange(len(nusc_obj.scene)):
        nuscData = NuscData(nusc_obj, i)
        mapName = nuscData.get_map()
        if mapName not in maps:
            nuscMap = NuScenesMap(
                dataroot=f"data/nuscenes/{args.version}",
                map_name=mapName,
            )
            maps[mapName] = nuscMap
        else:
            nuscMap = maps[mapName]
        nuscs.append((nuscData, nuscMap))
    pickles = glob.glob(
        f"./{os.path.join(args.dir,args.version)}/**/*.pickle", recursive=True
    )
    parmas = list()
    for path in pickles:
        out = os.path.join(out_dir, path.replace("/", "_")[12:-7])
        if not os.path.exists(f"{out}.mp4"):
            parmas.append((path, out_dir, nuscs))
    print(f"Total: {len(parmas)}")

    # for param in parmas:
    #     show(*param)
    sem = Semaphore(10)
    plist = list()
    for param in parmas:
        p = Process(target=sem_show, args=(sem, *param))
        p.start()
        plist.append(p)
    for p in plist:
        p.join()
    print("Done")


if __name__ == "__main__":
    main()
