from Data import Data
from Datalist import Datalist
from collections import defaultdict


class Dataset:
    def __init__(self, scene: str, inst: str) -> None:
        self.scene = scene
        self.inst = inst
        self.ego: Datalist = list()
        self.npc: Datalist = list()
        self.atk: Datalist = list()

    def set_ego(self, ego: Datalist) -> None:
        self.ego: Datalist = ego
        self.compile()

    def set_npc(self, npc: Datalist) -> None:
        self.npc: Datalist = npc
        self.compile()

    def set_atk(self, atk: Datalist) -> None:
        self.atk: Datalist = atk
        self.compile()

    def compile(self) -> None:
        self.get_timelist()
        self.gen_time2data()
        # print(self.time2data.keys())

    def gen_time2data(self) -> None:
        ret = defaultdict(dict)
        for d in self.ego:
            ret[d.timestamp]["ego"] = d
        for d in self.npc:
            ret[d.timestamp]["npc"] = d
        for d in self.atk:
            ret[d.timestamp]["atk"] = d
        self.time2data: dict[int, dict[str, Data]] = ret

    def get_timelist(self) -> None:
        ret = set()
        for d in self.ego:
            ret.add(d.timestamp)
        for d in self.npc:
            ret.add(d.timestamp)
        ret = list(ret)
        ret = sorted(ret)
        self.timelist = ret

    def filter(self) -> bool:
        self.atk.compile()
        maxv = max([t.velocity for t in self.atk])
        maxa = max([t.accelerate for t in self.atk])
        if maxv < 25 and maxa < 10:
            return True
        # print('filtered', self.scene['token'], self.inst['token'], maxv, maxa)
        return False
