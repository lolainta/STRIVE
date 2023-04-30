from numpy import cos, sin, pi, deg2rad
from generation.Translation import Translation


class Data:
    def __init__(self, ts, trans) -> None:
        self.timestamp = ts
        self.transform = trans
        self.width = 2
        self.length = 4
        self.bound = self.get_bound()

    def __sub__(self, o):
        # print(f'data  __sub__')
        # print(tf)
        ret = Data(self.timestamp - o.timestamp, self.transform - o.transform)
        # print('data __sub done')
        return ret

    def set_velocity(self, v: float) -> None:
        self.velocity = v

    def set_accelerate(self, a: float) -> None:
        self.accelerate = a

    def get_bound(self):
        x = self.transform.translation.x
        y = self.transform.translation.y
        yaw = self.transform.rotation.yaw
        width = self.width
        length = self.length
        tr = Translation(
            [
                x + length / 2 * cos(yaw) + width / 2 * cos(yaw - pi / 2),
                y + length / 2 * sin(yaw) + width / 2 * sin(yaw - pi / 2),
                0,
            ]
        )
        tl = Translation(
            [
                x + length / 2 * cos(yaw) - width / 2 * cos(yaw - pi / 2),
                y + length / 2 * sin(yaw) - width / 2 * sin(yaw - pi / 2),
                0,
            ]
        )
        br = Translation(
            [
                x - length / 2 * cos(yaw) + width / 2 * cos(yaw - pi / 2),
                y - length / 2 * sin(yaw) + width / 2 * sin(yaw - pi / 2),
                0,
            ]
        )
        bl = Translation(
            [
                x - length / 2 * cos(yaw) - width / 2 * cos(yaw - pi / 2),
                y - length / 2 * sin(yaw) - width / 2 * sin(yaw - pi / 2),
                0,
            ]
        )
        return [tr, tl, bl, br]

    def flip(self, side=0) -> None:
        self.transform.move(self.width, -pi / 2)
        self.bound = self.get_bound()

    def move(self, dis: float, deg: float) -> None:
        rad = deg2rad(deg)
        self.transform.move(dis, rad)
        self.bound = self.get_bound()

    def rotate(self, deg: float, org=None) -> None:
        self.transform.rotate(deg, org)
        self.bound = self.get_bound()
