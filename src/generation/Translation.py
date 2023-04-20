from numpy import sqrt


class Translation:
    def __init__(self, trans) -> None:
        if isinstance(trans, list):
            # print('list', trans)
            self.set_translation(trans)
        elif isinstance(trans, Translation):
            # print('translation', trans)
            # self = deepcopy(trans)
            assert False
        else:
            assert False, "Translation construct failed "

    def length(self) -> float:
        return sqrt(self.x**2 + self.y**2 + self.z**2)

    def set_translation(self, trans: list) -> None:
        self.raw_trans = trans
        self.x = trans[0]
        self.y = trans[1]
        self.z = trans[2]

    def lmul22(self, mat22):
        return Translation(
            [
                (self.x * mat22[0][0]) + (self.y * mat22[0][1]),
                (self.x * mat22[1][0]) + (self.y * mat22[1][1]),
                self.z,
            ]
        )

    def __add__(self, o):
        return Translation([self.x + o.x, self.y + o.y, self.z + o.z])

    def __sub__(self, o):
        ret = Translation([self.x - o.x, self.y - o.y, self.z - o.z])
        return ret

    # def __mul__(self, o):
    #     return Translation([self.x*o, self.y*o, self.z*o])

    # def __rmul__(self, o):
    #     return Translation([self.x*o, self.y*o, self.z*o])

    def __repr__(self) -> str:
        return f"Translation: [{self.x}, {self.y}, {self.z}]"
