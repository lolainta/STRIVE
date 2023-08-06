import argparse
import os
import pickle
from multiprocessing import Process, Semaphore
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from generation.Generator import Generator
from generation.NuscData import NuscData
from tqdm import trange


def gen_scene(gen: Generator, map: NuScenesMap, args):
    gen.verbose = True
    gen.fetch_data()
    dataCluster = gen.gen_all()
    validData = gen.filter_by_vel_acc(dataCluster)
    validData = gen.filter_by_curvature(validData)
    validData = gen.filter_by_collision(validData)
    validData = gen.filter_by_map(validData, map)
    osz = len(dataCluster)
    sz = len(validData)
    fsz = osz - sz
    if args.verbose:
        print(f"scene[{gen.nuscData.scene_id}] {osz}-{fsz}={sz} data generated")
    if args.record != "" and sz > 0:
        for idx, dataset in enumerate(validData):
            out_dir = os.path.join(
                args.record,
                args.dataset.split("-")[-1],
                dataset.cond.name,
                gen.nuscData.scene["name"],
            )
            fname = f"{dataset.idx}_{dataset.inst['token']}.pickle"
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, fname), "wb") as f:
                pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        if args.verbose:
            print(f"scene[{gen.nuscData.scene_id}] {sz} data recorded")


def generate(sem: Semaphore, gen: Generator, map: NuScenesMap, args):
    sem.acquire()
    print(f"scene[{gen.nuscData.scene_id}] Start")
    gen_scene(gen, map, args)
    print(f"scene[{gen.nuscData.scene_id}] Done")
    os.close(
        os.open(
            os.path.join(
                args.record, args.dataset.split("-")[-1], gen.nuscData.scene["name"]
            ),
            os.O_RDONLY | os.O_CREAT,
        )
    )
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
        if args.record != "" and os.path.exists(
            os.path.join(
                args.record, args.dataset.split("-")[-1], nusc.scene[i]["name"]
            )
        ):
            print(f"scene[{i}] already generated")
            continue
        nuscData = NuscData(nusc, i)
        mapName = nuscData.get_map()

        if mapName != "singapore-onenorth":
            continue

        if mapName not in maps:
            nuscMap = NuScenesMap(
                dataroot=f"data/nuscenes/{args.dataset.split('-')[-1]}",
                map_name=mapName,
            )
            maps[mapName] = nuscMap
        else:
            nuscMap = maps[mapName]
        nuscs.append((nuscData, nuscMap))

    print(f"{len(nuscs)} scenes to generate")
    os.makedirs(os.path.join(args.record, args.dataset.split("-")[-1]), exist_ok=True)
    plist = list()
    sem = Semaphore(10)
    for data, map in nuscs:
        gen: Generator = Generator(data)
        p = Process(target=generate, args=(sem, gen, map, args))
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
