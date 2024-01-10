from generation.Data import Data
from numpy import cos, sin, gradient
from csv import writer


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
        self.grad()

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

    def grad(self) -> None:
        xs = [d.transform.translation.x for d in self.datalist]
        ys = [d.transform.translation.y for d in self.datalist]
        zs = [d.transform.translation.z for d in self.datalist]
        if len(xs) == 0:
            dx = []
            dy = []
            dz = []
            ddx = []
            ddy = []
            ddz = []
        elif len(xs) == 1:
            dx = [0]
            dy = [0]
            dz = [0]
            ddx = [0]
            ddy = [0]
            ddz = [0]
        else:
            timestamps = [d.timestamp / 1e6 for d in self.datalist]
            dx = gradient(xs, timestamps)
            dy = gradient(ys, timestamps)
            dz = gradient(zs, timestamps)
            ddx = gradient(dx, timestamps)
            ddy = gradient(dy, timestamps)
            ddz = gradient(dz, timestamps)
        for i in range(len(self.datalist)):
            self.datalist[i].set_velocity((dx[i] ** 2 + dy[i] ** 2 + dz[i] ** 2) ** 0.5)
            self.datalist[i].set_accelerate(
                (ddx[i] ** 2 + ddy[i] ** 2 + ddz[i] ** 2) ** 0.5
            )
        return

    def get_max_curvature(self, step: int = 1) -> float:
        max_curvature = 0
        for i in range(len(self.datalist) - step):
            cur = self.datalist[i].transform
            new = self.datalist[i + step].transform
            diff = new - cur
            curvature = abs(diff.rotation.yaw) / diff.translation.length()
            max_curvature = max(max_curvature, curvature)
        return max_curvature

    def get_max_rotation(self) -> float:
        max_rotation = 0
        for i in range(len(self.datalist) - 1):
            cur = self.datalist[i].transform
            new = self.datalist[i + 1].transform
            diff = new - cur
            max_rotation = max(max_rotation, abs(diff.rotation.yaw))
        return max_rotation

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

    def export(self, wrt: writer, identity: str) -> None:
        self.grad()
        for d in self.datalist:
            wrt.writerow(
                [
                    d.timestamp,
                    identity,
                    d.transform.translation.x,
                    d.transform.translation.y,
                    d.velocity,
                    d.transform.rotation.yaw,
                ]
            )
