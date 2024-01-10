from generation.Data import Data
from generation.Datalist import Datalist
from generation.Condition import Condition
from nuscenes.map_expansion.map_api import NuScenesMap
import csv
from numpy import pi, rad2deg
from icecream import ic


class ColDataset:
    def __init__(self, scene: str, inst: dict, cond: Condition) -> None:
        self.scene = scene
        self.inst = inst
        self.ego: Datalist
        self.npcs: list[Datalist] = list()
        self.npc_tks: list[str] = list()
        self.atk: Datalist
        self.cond: Condition = cond
        self.idx = None
        self.filtered = False

    def set_ego(self, ego: Datalist) -> None:
        self.ego: Datalist = ego

    def add_npc(self, npc: Datalist, tk: str) -> None:
        self.npcs.append(npc)
        self.npc_tks.append(tk)

    def set_atk(self, atk: Datalist) -> None:
        self.atk: Datalist = atk

    def trim(self, timelist: list) -> None:
        self.ego.trim(timelist)
        for npc in self.npcs:
            npc.trim(timelist)
        self.atk.trim(timelist)
        self.timelist = timelist

    def filter(self) -> bool:
        if not self.filter_by_vel_acc():
            return False
        if not self.filter_by_rotation(pi / 2):
            return False
        if not self.filter_by_curvature(3):
            return False
        if not self.filter_by_step_curvature(3):
            return False
        if not self.filter_by_collision():
            return False
        return True

    def filter_by_vel_acc(self) -> bool:
        self.atk.compile()
        maxv = max([t.velocity for t in self.atk])
        maxa = max([t.accelerate for t in self.atk])
        if maxv < 25 and maxa < 10:
            self.filtered = True
            return True
        # print('filtered', self.scene['token'], self.inst['token'], maxv, maxa)
        return False

    def filter_by_rotation(self, threshold: float) -> bool:
        if self.atk.get_max_curvature() > threshold:
            self.filtered = True
            return False
        return True

    def filter_by_collision(self) -> bool:
        blocks = dict()
        for d in self.ego[:-1]:
            d: Data
            if d.timestamp not in blocks:
                blocks[d.timestamp] = list()
            blocks[d.timestamp].append(d.get_poly_bound())
        for npc in self.npcs:
            for d in npc:
                d: Data
                if d.timestamp not in blocks:
                    blocks[d.timestamp] = list()
                    # assert d.timestamp == self.ego[-1].timestamp
                blocks[d.timestamp].append(d.get_poly_bound())
        for d in self.atk:
            d: Data
            if d.timestamp not in blocks:
                continue
            for poly in blocks[d.timestamp]:
                if d.check_collision(poly):
                    self.filtered = True
                    return False
        return True

    def filter_by_curvature(self, threshold: int) -> bool:
        if self.atk.get_max_curvature() > threshold:
            self.filtered = True
            return False
        return True

    def filter_by_step_curvature(self, threshold: int) -> bool:
        if self.atk.get_max_curvature(step=2) > threshold:
            self.filtered = True
            return False
        return True

    def filter_by_map(self, map: NuScenesMap) -> bool:
        if not self.filter_side_intersection(map):
            return False
        if not self.filter_by_drivable_area(map):
            return False
        return True

    def filter_by_drivable_area(self, map: NuScenesMap) -> bool:
        for d in self.atk:
            cord = d.transform.translation
            x, y = cord.x, cord.y
            datk = map.record_on_point(x, y, "drivable_area")
            if datk == "":
                self.filtered = True
                return False
        return True

    def filter_side_intersection(self, map: NuScenesMap) -> bool:
        if self.cond != Condition.JC or self.cond != Condition.LTAP:
            return True
        d = self.atk[-1]
        cord = d.transform.translation
        x, y = cord.x, cord.y

        def is_intersection(x, y):
            rstk = map.record_on_point(x, y, "road_segment")
            if rstk == "":
                return False
            rs = map.get("road_segment", rstk)
            return rs["is_intersection"]

        if not is_intersection(x, y):
            print(f"JC not in intersection: {self.scene['name']} {self.inst['token']}")
            self.filtered = True
            return False
        return True

    def export(self, path: str) -> None:
        with open(path, "w", newline="\n") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow(["TIMESTAMP", "TRACK_ID", "X", "Y", "V", "YAW"])
            self.ego.export(writer, "ego")
            self.atk.export(writer, self.inst["token"])
            for npc_tk, npc in zip(self.npc_tks, self.npcs):
                if npc_tk == self.inst["token"]:
                    continue
                npc.export(writer, npc_tk)
