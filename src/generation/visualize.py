import argparse
import os
import pickle
import random
from Drawer import Drawer


def gen_random() -> str:
    scenes = os.listdir("./records")
    scene = random.sample(scenes, 1)[0]
    insts = os.listdir(os.path.join("./records", scene))
    inst = random.sample(insts, 1)[0]
    record = os.path.join("./records", scene, inst)
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
        default=gen_random(),
        help="Dataset folder",
    )
    parser.add_argument("--one", action="store_true")
    args = parser.parse_args()
    print(args)
    if args.one:
        show(args.file)
    else:
        while True:
            show(gen_random())


if __name__ == "__main__":
    main()
