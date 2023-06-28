import argparse
import pickle
from generation.Drawer import Drawer
from generation.NuscData import NuscData
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
import glob, os
from multiprocessing import Process,Semaphore
from tqdm import trange

def show(path: str, nuscs, args):
    print(f"Loading {path}")
    with open(path, "rb") as f:
        dataset = pickle.load(f)
    found = False
    for nuscData, nuscMap in nuscs:
        if nuscData.scene["name"] == dataset.scene["name"]:
            found = True
            out = os.path.join(args.out, path.replace("/", "_")[12:-7])
            if os.path.exists(f"{out}.mp4"):
                print(f"Skip: {path}", flush=True)
                continue
            plt = Drawer(nuscData, nuscMap)
            plt.plot_dataset(dataset, out)
            plt.close()
            print(f"Saved: {path} {dataset.inst['token']}", flush=True)
    if not found:
        assert False, f"Scene {dataset.scene['name']} not found"

def sem_show(sem:Semaphore, path: str, nuscs, args):
    sem.acquire()
    show(path, nuscs, args)
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
        "-o",
        "--out",
        required=True,
        default=None,
        help="Output folder",
    )
    args = parser.parse_args()
    print(args)
    nusc_obj = NuScenes(
        version="v1.0-trainval", dataroot="data/nuscenes/trainval", verbose=True
    )
    maps = dict()
    nuscs = list()
    for i in trange(len(nusc_obj.scene)):
        nuscData = NuscData(nusc_obj, i)
        mapName = nuscData.get_map()
        if mapName not in maps:
            nuscMap = NuScenesMap(
                dataroot="data/nuscenes/trainval",
                map_name=mapName,
            )
            maps[mapName] = nuscMap
        else:
            nuscMap = maps[mapName]            
        nuscs.append((nuscData, nuscMap))
    pickles = glob.glob(f"./{args.dir}/**/*.pickle", recursive=True)
    pickles.sort()
    plist = list()
    sem = Semaphore(10)
    for path in pickles:
        p = Process(target=sem_show, args=(sem, path, nuscs, args))
        p.start()
        plist.append(p)
    for p in plist:
        p.join()
    print("Done")

if __name__ == "__main__":
    main()
