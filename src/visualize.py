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


def show(path: str, nuscs, args):
    out = os.path.join(args.out, path.replace("/", "_")[12:-7])
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


def sem_show(sem: Semaphore, path: str, nuscs, args):
    sem.acquire()
    attempt = 0
    while True:
        attempt += 1
        try:
            if attempt > 10:
                assert False, f"Too many attempts: {path}"
            show(path, nuscs, args)
        except AssertionError as e:
            print(f"Error: {path} {e}", flush=True)
            break
        except Exception as e:
            print(f"Error: {path} {e}", flush=True)
            sleep(1)
            print(f"Retrying: {path}", flush=True)
            continue
        else:
            break
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
    pickles = glob.glob(f"./{args.dir}/**/*.pickle", recursive=True)
    parmas = list()
    for path in pickles:
        out = os.path.join(args.out, path.replace("/", "_")[12:-7])
        if not os.path.exists(f"{out}.mp4"):
            parmas.append((path, nuscs, args))
    print(f"Total: {len(parmas)}")
    # for param in parmas:
    #     show(*param)
    # pickles.sort()
    # for path in pickles:
    #     show(path, nuscs, args)
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
