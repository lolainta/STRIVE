from generation.Data import Data
from generation.Transform import Transform
from numpy import cos, sin


class Datalist:
    def __init__(self) -> None:
        self.datalist: list[Data] = list()

    def __getitem__(self, key: int) -> Data:
        return self.datalist[key]

    def __len__(self):
        return len(self.datalist)

    def append(self, d: Data):
        self.datalist.append(d)

    def compile(self) -> None:
        self.elapse_time = (self[-1].timestamp - self[0].timestamp) / 1e6
        self.gen_velocity()
        self.gen_accelerate()

    def trim(self, timelist: list) -> None:
        newlist = list()
        for data in self.datalist:
            if data.timestamp in timelist:
                newlist.append(data)
        self.datalist = newlist

    def gen_timelist(self) -> list:
        ret = list()
        for data in self.datalist:
            ret.append(data.timestamp)
        self.timelist = ret
        return ret

    def gen_velocity(self) -> None:
        dxdt = list()
        for i in range(len(self.datalist) - 1):
            sub: Data = self.datalist[i + 1] - self.datalist[i]
            dis = sub.transform.length()
            dxdt.append(dis / sub.timestamp * 1e6)
        self.datalist[0].set_velocity(dxdt[0])
        self.datalist[-1].set_velocity(dxdt[-1])
        for i in range(1, len(self.datalist) - 1):
            self.datalist[i].set_velocity((dxdt[i - 1] + dxdt[i]) / 2)

    def gen_accelerate(self) -> None:
        dvdt = list()
        for i in range(len(self.datalist) - 1):
            sub: float = self.datalist[i + 1].velocity - self.datalist[i].velocity
            dt = self.datalist[i + 1].timestamp - self.datalist[i].timestamp
            dvdt.append(sub / dt * 1e6)
        self.datalist[0].set_accelerate(dvdt[0])
        self.datalist[-1].set_accelerate(dvdt[-1])
        for i in range(1, len(self.datalist) - 1):
            self.datalist[i].set_accelerate((dvdt[i - 1] + dvdt[i]) / 2)

    def get_max_curvature(self) -> float:
        max_curvature = 0
        for i in range(len(self.datalist) - 1):
            cur = self.datalist[i].transform
            new = self.datalist[i + 1].transform
            diff = new - cur
            curvature = abs(diff.rotation.yaw) / diff.translation.length()
            max_curvature = max(max_curvature, curvature)
        return max_curvature

    def serialize(self) -> list:
        ret = list()
        for data in self.datalist:
            cur = dict()
            cur["x"] = data.transform.translation.x
            cur["y"] = data.transform.translation.y
            cur["h"] = data.transform.rotation.yaw
            cur["hcos"] = cos(cur["h"])
            cur["hsin"] = sin(cur["h"])
            cur["t"] = data.timestamp
            cur["samp_tok"] = "XXX"
            ret.append(cur)
        return ret
