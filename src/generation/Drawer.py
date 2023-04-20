import matplotlib.pyplot as plt
import numpy as np
import sys
from Data import Data
from Dataset import Dataset
from Translation import Translation


class Drawer:
    def __init__(self, delay=1e-12) -> None:
        # plt.gcf().canvas.mpl_connect('key_release_event',
        #                              lambda event: [exit(0) if event.key == 'escape' else None])
        self.fig, self.ax = plt.subplots(figsize=(20, 20))
        self.delay = delay
        self.ax.set_xticks(range(1720, 1723), fontsize=20)
        self.ax.set_yticks(range(2668, 2672), fontsize=20)

    def plot_arrow(self, x, y, yaw, length=2.0, width=1, fc="r", ec="k") -> None:
        self.ax.arrow(
            x,
            y,
            length * np.cos(yaw),
            length * np.sin(yaw),
            fc=fc,
            ec=ec,
            head_width=width,
            head_length=width,
        )
        self.ax.plot(x, y)

    def plot_seg(self, p1: Translation, p2: Translation, col="green") -> None:
        self.ax.plot([p1.x, p2.x], [p1.y, p2.y], "o--", color=col)
        self.ax.scatter(p1.x, p1.y, c=col)
        self.ax.scatter(p2.x, p2.y, c=col)

    def plot_box(self, bound, col="green") -> None:
        for i in range(4):
            self.plot_seg(bound[i], bound[(i + 1) % 4], col=col)

    def plot_car(self, d: Data, col="green") -> None:
        x = d.transform.translation.x
        y = d.transform.translation.y
        yaw = d.transform.rotation.yaw
        bnd = d.bound
        self.plot_box(bnd, col=col)
        self.plot_arrow(x, y, yaw, fc=col)

    def plot_dataset(self, ds: Dataset, atk=False) -> None:
        print(
            f'Drawing dataset: scene={ds.scene["token"]} inst={ds.inst["token"]}',
            file=sys.stderr,
        )
        for v in ds.time2data.values():
            plt.cla()
            if "ego" in v:
                self.plot_car(v["ego"], col="blue")
            self.plot_car(ds.ego[0], col="blue")
            self.plot_car(ds.ego[-1], col="blue")
            if atk:
                self.plot_car(ds.atk[0])
                self.plot_car(ds.atk[-1])
                if "atk" in v:
                    self.plot_car(v["atk"])
            else:
                self.plot_car(ds.npc[0])
                self.plot_car(ds.npc[-1])
                if "npc" in v:
                    self.plot_car(v["npc"])
            if self.delay:
                plt.pause(self.delay)

    def show(self) -> None:
        plt.show()

    def close(self) -> None:
        plt.close()
