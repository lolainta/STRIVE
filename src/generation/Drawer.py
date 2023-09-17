import matplotlib.pyplot as plt
import numpy as np
import sys
from generation.Data import Data
from generation.Dataset import ColDataset
from generation.Translation import Translation
from nuscenes.map_expansion.map_api import NuScenesMap
import os
import warnings


class Drawer:
    def __init__(self, nuscMap: NuScenesMap, delay=1e-12) -> None:
        self.delay = delay
        self.nuscMap = nuscMap

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
        self.ax.plot([p1.x, p2.x], [p1.y, p2.y], "-", color=col)

    def plot_box(self, bound, col="green") -> None:
        for i in range(4):
            self.plot_seg(bound[i], bound[(i + 1) % 4], col=col)

    def plot_car(self, d: Data, col="green", no_box=False) -> None:
        x = d.transform.translation.x
        y = d.transform.translation.y
        yaw = d.transform.rotation.yaw
        bnd = d.bound
        if not no_box:
            self.plot_box(bnd, col=col)
        self.plot_arrow(x, y, yaw, fc=col)

    def plot_dataset(self, ds: ColDataset, out: str) -> None:
        print(
            f"Drawing dataset: {ds.cond.name} {ds.scene['name']} {ds.idx} {ds.inst['token']}",
            file=sys.stderr,
        )
        os.makedirs(out, exist_ok=True)
        for idx, cur_time in enumerate(ds.timelist):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.fig, self.ax = self.nuscMap.render_layers(["drivable_area"])
            center = ds.ego.datalist[idx].transform.translation
            self.ax.set_xlim(center.x - 100, center.x + 100)
            self.ax.set_ylim(center.y - 100, center.y + 100)
            assert ds.ego.datalist[idx].timestamp == cur_time
            self.plot_car(ds.ego.datalist[idx], col="blue")
            for npc in ds.npcs:
                for npc_data in npc.datalist:
                    if npc_data.timestamp == cur_time:
                        self.plot_car(npc_data, col="green")
            for atk_data in ds.atk.datalist:
                if atk_data.timestamp == cur_time:
                    self.plot_car(atk_data, col="red")
            frame = os.path.join(out, f"{idx:02d}.png")
            plt.savefig(frame)
            plt.cla()
            plt.clf()
            plt.close()

    def plot_atks(self, atks: list, out: str, no_box=False) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.fig, self.ax = self.nuscMap.render_layers(["drivable_area"])
        for atk, col in atks:
            self.plot_car(atk[-1], col=col, no_box=no_box)
        # xmin = min([atk[-1].transform.translation.x for atk, _ in atks])
        # xmax = max([atk[-1].transform.translation.x for atk, _ in atks])
        # ymin = min([atk[-1].transform.translation.y for atk, _ in atks])
        # ymax = max([atk[-1].transform.translation.y for atk, _ in atks])
        # self.ax.set_xlim(xmin - 20, xmax + 20)
        # self.ax.set_ylim(ymin - 20, ymax + 20)
        plt.savefig(out)
        plt.cla()
        plt.clf()
        plt.close()

    def export_mp4(self, out: str, h264=True) -> None:
        if h264:
            os.system(
                f"ffmpeg -r 2 -i {os.path.join(out,'%02d.png')} -pix_fmt yuv420p -y {out}.mp4 -v quiet"
            )
        else:
            os.system(
                f"ffmpeg -r 2 -i {os.path.join(out,'%02d.png')} -y {out}.mp4 -v quiet"
            )

    def show(self) -> None:
        plt.show()

    def close(self) -> None:
        plt.close()
