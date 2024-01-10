import argparse
import pickle
from generation.Drawer import Drawer
from generation.NuscData import NuscData
from generation.Dataset import ColDataset
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
import glob, os
from multiprocessing import Process, Semaphore
from tqdm import trange, tqdm
import random
from icecream import ic


def parse_cfg():
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
        default="trainval",
        choices=["mini", "trainval", "middle"],
        help="Data version",
    )
    parser.add_argument(
        "-e",
        "--encode",
        action="store_true",
        help="Encode with H.264",
    )
    args = parser.parse_args()
    print(args)
    return args


def load_data(path: str) -> tuple:
    with open(path, "rb") as f:
        dataset = pickle.load(f)
    return dataset


def filter(dataset: ColDataset, nuscs) -> bool:
    if not dataset.filter():
        return False
    for nuscData, nuscMap in nuscs:
        if nuscData.scene["name"] == dataset.scene["name"]:
            if not dataset.filter_by_map(nuscMap):
                return False
            break
    else:
        assert False, f"Scene {dataset.scene['name']} not found"
    return True


def show(dataset: ColDataset, nuscs, encode: bool, out):
    for nuscData, nuscMap in nuscs:
        if nuscData.scene["name"] == dataset.scene["name"]:
            plt = Drawer(nuscMap)
            plt.plot_dataset(dataset, out)
            plt.export_mp4(out, encode)
            plt.close()
            break
    else:
        assert False, f"Scene {dataset.scene['name']} not found"


def sem_show(sem: Semaphore, path: str, out_dir, nuscs, encode: bool):
    sem.acquire()
    if not os.path.exists(f"{out_dir}/{path.replace('/', '_')[12:-7]}.mp4"):
        # if random.randint(0, 100) == 0:
        #     tqdm.write(f"Loading: {path}", flush=True)
        dataset: ColDataset = load_data(path)
        if not filter(dataset, nuscs):
            tqdm.write(f"Filtered: {path}", flush=True)
            os.remove(path)
            sem.release()
            return
        # out = os.path.join(out_dir, path.replace("/", "_")[12:-7])
        # show(dataset, nuscs, encode, out)
        # print(f"Done: {path}", flush=True)
    else:
        # assert False, f"Skipped: {path}"
        # if random.randint(0, 100) == 0:
        #     tqdm.write(f"Skipped: {path}", flush=True)
        pass
    sem.release()


def main():
    args = parse_cfg()
    out_dir = os.path.join(args.dir, args.version, "viz")
    maps = dict()
    nuscs = list()
    nusc_obj = NuScenes(
        version=f"v1.0-{'mini' if args.version == 'mini' else 'trainval'}",
        dataroot=f"data/nuscenes/{'mini' if args.version == 'mini' else 'trainval'}",
        verbose=True,
    )
    for i in trange(len(nusc_obj.scene)):
        nuscData = NuscData(nusc_obj, i)
        mapName = nuscData.get_map()
        if mapName not in maps:
            nuscMap = NuScenesMap(
                dataroot=f"data/nuscenes/{'mini' if args.version == 'mini' else 'trainval'}",
                map_name=mapName,
            )
            maps[mapName] = nuscMap
        else:
            nuscMap = maps[mapName]
        # nuscMap = NuScenesMap(
        #     dataroot=f"data/nuscenes/{'mini' if args.version == 'mini' else 'trainval'}",
        #     map_name=mapName,
        # )
        nuscs.append((nuscData, nuscMap))
    pickles = glob.glob(
        f"./{os.path.join(args.dir,args.version)}/**/*.pickle", recursive=True
    )
    # pickles = ['./data/hcis/v4.1/trainval/boston-seaport/HO/scene-0164/5-0-29_83515afd82064e64b301d56843858d7e.pickle']
    pickles = [p for p in pickles if p.find("scene-0655") != -1]
    # ic(pickles)
    random.shuffle(pickles)
    params = list()
    for path in pickles:
        params.append((path, out_dir, nuscs, args.encode))
    print(f"Total: {len(params)}")
    # params = random.sample(params, min(len(params), 40))
    print(f"Sampled: {len(params)}")
    sem = Semaphore(200)
    for param in tqdm(params):
        sem_show(sem, *param)
    # plist = list()
    # for param in params:
    #     p = Process(target=sem_show, args=(sem, *param))
    #     p.start()
    #     plist.append(p)
    # for p in plist:
    #     p.join()
    print("Done")


if __name__ == "__main__":
    main()
