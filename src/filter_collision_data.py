import argparse
import pickle
from generation.Dataset import ColDataset
import os
from tqdm import tqdm, trange
import glob


def parse_cfg():
    parser = argparse.ArgumentParser(
        prog="python3 src/filter_collision_data.py",
        description="Filter generated dataset from given pickle file",
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
        help="Dataset version, mini or trainval",
    )
    args = parser.parse_args()
    print(args)
    return args


def filter_dir(scene_dir: str) -> tuple:
    # tqdm.write(f"Filtering: {scene_dir}")
    cur, total = 0, 0
    for file in os.listdir(scene_dir):
        with open(os.path.join(scene_dir, file), "rb") as f:
            dataset: ColDataset = pickle.load(f)
        # tqdm.write(f"Filtering: {dataset.scene['name']} {file}")
        total += 1
        if not dataset.filter():
            cur += 1
            os.remove(os.path.join(scene_dir, file))
        # with open(os.path.join(scene_dir, file), "wb") as f:
        #     pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    if cur != 0:
        tqdm.write(f"Filtered: {cur} out of {total} from {scene_dir}")
    return cur, total


def bind_nusc_with_map(version):
    from nuscenes.nuscenes import NuScenes
    from nuscenes.map_expansion.map_api import NuScenesMap
    from generation.NuscData import NuscData

    maps = dict()
    nuscs = list()
    nusc_obj = NuScenes(
        version=f"v1.0-{'mini' if version == 'mini' else 'trainval'}",
        dataroot=f"data/nuscenes/{'mini' if version == 'mini' else 'trainval'}",
        verbose=True,
    )
    for i in trange(len(nusc_obj.scene)):
        nuscData = NuscData(nusc_obj, i)
        mapName = nuscData.get_map()
        if mapName not in maps:
            nuscMap = NuScenesMap(
                dataroot=f"data/nuscenes/{'mini' if version == 'mini' else 'trainval'}",
                map_name=mapName,
            )
            maps[mapName] = nuscMap
        else:
            nuscMap = maps[mapName]
        nuscs.append((nuscData, nuscMap))
    return nuscs


def main():
    args = parse_cfg()
    nusc = bind_nusc_with_map(args.version)
    pickles = glob.glob(os.path.join(args.dir, "**/*.pickle"), recursive=True)
    print(f"Total: {len(pickles)}")
    cnt, total = 0, 1e-6
    pbar = tqdm(total=len(pickles))
    for root, dir, files in os.walk(args.dir):
        files.sort()
        if len(files) and all(f[-7:] == (".pickle") for f in files):
            cur, cdir = filter_dir(root)
            cnt += cur
            total += cdir
            assert cdir == len(files), f"{cdir} {len(files)}"
            pbar.update(len(files))
    print(f"Filtered: {cnt} out of {int(total)} from {args.dir}. Ratio {cnt/total}")


if __name__ == "__main__":
    main()
