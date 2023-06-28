import argparse
import os
import pickle
from multiprocessing import Process, Semaphore
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from generation.Generator import Generator, Condition
from generation.NuscData import NuscData
from tqdm import trange


def gen_by_cond(gen: Generator, type: Condition, nuscMap: NuScenesMap, args):
    scene_dir = os.path.join(
        args.record,
        args.dataset.split("-")[-1],
        type.name,
        gen.nuscData.scene["name"],
    )
    if os.path.exists(scene_dir):
        print(f"scene[{gen.nuscData.scene_id}]/{type.name} already exists")
        return
    idx = gen.nuscData.scene_id
    dataCluster = gen.gen_all(type)
    validData = gen.filter_by_vel_acc(dataCluster)
    validData = gen.filter_by_map(dataCluster, nuscMap)
    osz = len(dataCluster)
    sz = len(validData)
    fsz = osz - sz
    if args.verbose:
        print(f"scene[{idx}]/{type.name} {osz}-{fsz}={sz} data generated")
    os.makedirs(scene_dir, exist_ok=True)
    if args.record != "" and sz > 0:
        print(f"scene[{idx}]/{type.name} {sz} data recorded")
        for idx, dataset in enumerate(validData):
            with open(os.path.join(scene_dir, f"{idx}.pickle"), "wb") as f:
                pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)


def gen_scene(gen: Generator, map: NuScenesMap, args):
    for cond in Condition:
        print(f"scene[{gen.nuscData.scene_id}]/{cond.name} Start")
        gen_by_cond(gen, cond, map, args)


def generte(sem: Semaphore, gen: Generator, map: NuScenesMap, args):
    sem.acquire()
    print(f"scene[{gen.nuscData.scene_id}] Start")
    gen_scene(gen, map, args)
    print(f"scene[{gen.nuscData.scene_id}] Done")
    sem.release()


def run(args):
    print("Loading Data...")
    nusc = NuScenes(
        version=args.dataset,
        dataroot=f"data/nuscenes/{args.dataset.split('-')[-1]}",
        verbose=args.verbose,
    )
    maps = dict()
    nuscs = list()
    for i in trange(len(nusc.scene)):
        nuscData = NuscData(nusc, i)
        mapName = nuscData.get_map()
        if mapName not in maps:
            nuscMap = NuScenesMap(
                dataroot=f"data/nuscenes/{args.dataset.split('-')[-1]}",
                map_name=mapName,
            )
            maps[mapName] = nuscMap
        else:
            nuscMap = maps[mapName]
        nuscs.append((nuscData, nuscMap))
    print("Data Loaded")
    plist = list()
    sem = Semaphore(10)
    for data, map in nuscs:
        gen: Generator = Generator(data)
        p = Process(target=generte, args=(sem, gen, map, args))
        p.start()
        plist.append(p)
    for p in plist:
        p.join()
    print("Done")


def main():
    parser = argparse.ArgumentParser(
        prog="python3 src/generation/main.py",
        description="Generate data from given nuscene dataset and collision type",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        choices=["v1.0-mini", "v1.0-trainval"],
        default="v1.0-mini",
        help="Nuscene dataset version",
    )
    parser.add_argument(
        "--record",
        default="",
        help="Specify the path to store the generated data (create if not exists)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Show log")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
