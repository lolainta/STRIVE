import argparse
import os
import pickle
import random
from generation.Drawer import Drawer


def gen_random(folder: str) -> str:
    scenes = os.listdir(folder)
    scene = random.sample(scenes, 1)[0]
    insts = os.listdir(os.path.join(folder, scene))
    inst = random.sample(insts, 1)[0]
    record = os.path.join(folder, scene, inst)
    return record


def show(file):
    print(f"Loading file: {file}")
    with open(file, "rb") as f:
        dataset = pickle.load(f)
    plt = Drawer()
    plt.plot_dataset(dataset)
    plt.plot_dataset(dataset, atk=True)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        prog="python3 src/visualize.py",
        description="Visualize generated dataset from given pickle file",
    )
    parser.add_argument(
        "-f",
        "--file",
        required=True,
        default=None,
        help="Dataset folder",
    )
    parser.add_argument("--one", action="store_true")
    args = parser.parse_args()
    print(args)
    if args.one:
        show(args.file)
    else:
        while True:
            show(gen_random(args.file))


if __name__ == "__main__":
    main()
