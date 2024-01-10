from pyquaternion import Quaternion
from numpy import arctan2, pi


class Rotation:
    def __init__(self, *args) -> None:
        assert len(args) == 1, "Rotation constructor failed"
        if isinstance(args[0], float):
            self.yaw = args[0]
        elif isinstance(args[0], list):
            self.set_rotation(args[0])
        elif isinstance(args[0], Rotation):
            self = args[0]
        else:
            assert False, "Rotation construct failed "

    def __sub__(self, o):
        assert -pi <= self.yaw <= pi, f"Rotation sub failed: {self.yaw}"
        assert -pi <= o.yaw <= pi, f"Rotation sub failed: {o.yaw}"
        ret = o.yaw - self.yaw
        if not -pi <= ret <= pi:
            if ret > pi:
                ret -= 2 * pi
            elif ret < -pi:
                ret += 2 * pi
        assert -pi <= ret <= pi, f"Rotation sub failed: {self.yaw}-{o.yaw}={ret}"
        return ret

    def set_rotation(self, rotation: list) -> None:
        self.raw_rotation = rotation
        rot = Quaternion(rotation).rotation_matrix
        rot = arctan2(rot[1, 0], rot[0, 0])
        self.yaw = rot

    def __repr__(self):
        return f"Rotation: yaw={self.yaw}"
