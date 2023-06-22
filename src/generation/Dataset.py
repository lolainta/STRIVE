from generation.Data import Data
from generation.Datalist import Datalist
from generation.Condition import Condition
from collections import defaultdict
from nuscenes.map_expansion.map_api import NuScenesMap


class ColDataset:
    def __init__(self, scene: str, inst: str, type: Condition) -> None:
        self.scene = scene
        self.inst = inst
        self.ego: Datalist = list()
        self.npcs: Datalist = list()
        self.atk: Datalist = list()
        self.type: Condition = type

    def set_ego(self, ego: Datalist) -> None:
        self.ego: Datalist = ego

    def add_npc(self, npc: Datalist) -> None:
        self.npcs.append(npc)

    def set_atk(self, atk: Datalist) -> None:
        self.atk: Datalist = atk

    # def compile(self) -> None:
    #     self.get_timelist()
    #     self.gen_time2data()
    #     # print(self.time2data.keys())

    # def gen_time2data(self) -> None:
    #     ret = defaultdict(dict)
    #     for d in self.ego:
    #         ret[d.timestamp]["ego"] = d
    #     for d in self.npc:
    #         ret[d.timestamp]["npc"] = d
    #     for d in self.atk:
    #         ret[d.timestamp]["atk"] = d
    #     self.time2data: dict[int, dict[str, Data]] = ret

    # def get_timelist(self) -> None:
    #     ret = set()
    #     for d in self.ego:
    #         ret.add(d.timestamp)
    #     for d in self.npc:
    #         ret.add(d.timestamp)
    #     ret = list(ret)
    #     ret = sorted(ret)
    #     self.timelist = ret

    def filter_by_vel_acc(self) -> bool:
        self.atk.compile()
        maxv = max([t.velocity for t in self.atk])
        maxa = max([t.accelerate for t in self.atk])
        if maxv < 25 and maxa < 10:
            return True
        # print('filtered', self.scene['token'], self.inst['token'], maxv, maxa)
        return False

    def filter_by_map(self, map: NuScenesMap) -> bool:
        for d in self.atk:
            cord = d.transform.translation
            x, y = cord.x, cord.y
            ls = map.layers_on_point(x, y)
            if ls["drivable_area"] == "":
                return False
        return True
