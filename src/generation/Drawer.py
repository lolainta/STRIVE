import matplotlib.pyplot as plt
import numpy as np
import sys
from generation.Data import Data
from generation.NuscData import NuscData
from generation.Dataset import ColDataset
from generation.Translation import Translation
from nuscenes.map_expansion.map_api import NuScenesMap
import os
import warnings


class Drawer:
    def __init__(self, nuscData: NuscData, nuscMap: NuScenesMap, delay=1e-12) -> None:
        self.delay = delay
        self.nusc = nuscData.nusc
        self.nuscData = nuscData
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

    def plot_car(self, d: Data, col="green") -> None:
        x = d.transform.translation.x
        y = d.transform.translation.y
        yaw = d.transform.rotation.yaw
        bnd = d.bound
        self.plot_box(bnd, col=col)
        self.plot_arrow(x, y, yaw, fc=col)

    def plot_dataset(self, ds: ColDataset, out: str) -> None:
        print(
            f'Drawing dataset: {ds.scene["name"]} inst={ds.inst["token"]}',
            file=sys.stderr,
        )
        os.makedirs(out, exist_ok=True)
        for idx, cur_time in enumerate(self.nuscData.times):
            plt.cla()
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
        os.system(
            f"ffmpeg -r 2 -pix_fmt yuv420p -i {os.path.join(out,'%02d.png')} -y {out}.mp4 > /dev/null 2>&1"
        )

    def show(self) -> None:
        plt.show()

    def close(self) -> None:
        plt.close()
