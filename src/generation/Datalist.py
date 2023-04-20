from Data import Data


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
