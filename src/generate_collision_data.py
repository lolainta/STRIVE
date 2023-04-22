import argparse
import os
import pickle
from nuscenes.nuscenes import NuScenes
from generation.Generator import Generator
from generation.NuscData import NuscData
from multiprocessing import Process


def gen_scene(nusc, idx, args):
    nuscData: NuscData = NuscData(nusc, idx)
    gen: Generator = Generator(nuscData)
    dataCluster = gen.gen_all()
    validData = [ds for ds in dataCluster if ds.filter() == True]
    osz = len(dataCluster)
    sz = len(validData)
    fsz = osz - sz
    if args.verbose:
        print(f"scene[{idx}]: {osz}-{fsz}={sz} data generated")
    if args.record and sz > 0:
        scene_dir = os.path.join(args.record_path, dataCluster[0].scene["name"])
        os.makedirs(scene_dir, exist_ok=True)
        for dataset in dataCluster:
            with open(
                os.path.join(scene_dir, f'{dataset.inst["token"]}.pickle'), "wb"
            ) as f:
                pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)


def run(args):
    print("Loading Data...")
    nusc = NuScenes(
        version=args.dataset,
        dataroot=f"data/nuscenes/{args.dataset.split('-')[-1]}",
        verbose=args.verbose,
    )
    plist = list()
    for i in range(len(nusc.scene)):
        p = Process(target=gen_scene, args=(nusc, i, args))
        plist.append(p)

    print("Generating Data...")
    for t in plist:
        t.start()
    for t in plist:
        t.join()
    print("======")


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
        "--record", action="store_true", help="Whether to record to record path or not)"
    )
    parser.add_argument(
        "--record-path",
        default="./records",
        help="The path to store the generated data (create if not exists)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Show log")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
