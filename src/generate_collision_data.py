import argparse
import os
import pickle
from multiprocessing import Process, Semaphore
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from generation.Generator import Generator
from generation.NuscData import NuscData
from generation.Dataset import ColDataset
from tqdm import trange


def parse_cfg():
    parser = argparse.ArgumentParser(
        prog="python3 src/generate_collision_data.py",
        description="Generate data from given nuscene dataset and collision type",
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
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=os.cpu_count(),
        help="Number of jobs to run in parallel",
    )
    args = parser.parse_args()
    print(args)
    return args


def gen_scene(gen: Generator, map: NuScenesMap, data_dir: str):
    gen.fetch_data()
    dataCluster = gen.gen_all(map)
    validData = gen.filter_by_map(dataCluster, map)
    osz = len(dataCluster)
    sz = len(validData)
    fsz = osz - sz
    print(f"scene[{gen.nuscData.scene_id}] {osz}-{fsz}={sz} data generated")
    for dataset in validData:
        dataset: ColDataset
        out_dir = os.path.join(
            data_dir,
            gen.nuscData.get_map(),
            dataset.cond.name,
            gen.nuscData.scene["name"],
        )
        fname = f"{dataset.idx}_{dataset.inst['token']}.pickle"
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, fname), "wb") as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"scene[{gen.nuscData.scene_id}] {sz} data recorded")


def generate(sem, gen: Generator, map: NuScenesMap, data_dir: str):
    sem.acquire()
    print(f"scene[{gen.nuscData.scene_id}] Start")
    gen_scene(gen, map, data_dir)
    print(f"scene[{gen.nuscData.scene_id}] Done")
    os.close(
        os.open(
            os.path.join(data_dir, "done", gen.nuscData.scene["name"]),
            os.O_RDONLY | os.O_CREAT,
        )
    )
    sem.release()


def run(args):
    print("Loading Data...")
    nusc = NuScenes(
        version=f"v1.0-{args.version}",
        dataroot=f"data/nuscenes/{args.version}",
        verbose=True,
    )
    maps = dict()
    nuscs = list()
    data_dir = os.path.join(args.dir, args.version)
    maps = [
        "singapore-onenorth",
        "singapore-hollandvillage",
        "singapore-queenstown",
        "boston-seaport",
    ]
    for i in trange(len(nusc.scene)):
        if os.path.exists(os.path.join(data_dir, "done", nusc.scene[i]["name"])):
            continue
        nuscData = NuscData(nusc, i)
        mapName = nuscData.get_map()
        assert mapName in maps, mapName
        nuscMap = NuScenesMap(
            dataroot=f"data/nuscenes/{args.version}",
            map_name=mapName,
        )
        nuscs.append((nuscData, nuscMap))
        if len(nuscs) >= 250:
            break

    print(f"{len(nuscs)} scenes to generate")
    os.makedirs(os.path.join(data_dir, "done"), exist_ok=True)
    plist = list()
    sem = Semaphore(args.jobs)
    for data, map in nuscs:
        gen: Generator = Generator(data)
        p = Process(target=generate, args=(sem, gen, map, data_dir))
        p.start()
        plist.append(p)
    for p in plist:
        p.join()
    print("Done")


def main():
    args = parse_cfg()
    run(args)


if __name__ == "__main__":
    main()
