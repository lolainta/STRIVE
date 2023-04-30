from generation.Rotation import Rotation
from generation.Translation import Translation
from numpy import cos, sin, deg2rad, pi


class Transform:
    def __init__(self, *args) -> None:
        if len(args) == 1:
            if isinstance(args[0], Transform):
                self = args[0]
            else:
                assert False, "Transform construct failed "
        elif len(args) == 2:
            if isinstance(args[0], Translation):
                self.translation: Translation = args[0]
            else:
                self.translation: Translation = Translation(args[0])
            if isinstance(args[1], Rotation):
                self.rotation: Rotation = args[1]
            else:
                self.rotation: Rotation = Rotation(args[1])
        else:
            assert False, "Transform construct failed "
        self.translation: Translation
        self.rotation: Rotation

    def __sub__(self, o):
        ret = Transform(self.translation - o.translation, self.rotation - o.rotation)
        return ret

    def __repr__(self):
        return f"Transform:\n\t{self.translation}\n\t{self.rotation}"

    def length(self) -> float:
        return self.translation.length()

    def move(self, dis, relate_dir) -> None:
        relate_dir += self.rotation.yaw
        self.translation.x += dis * cos(relate_dir)
        self.translation.y += dis * sin(relate_dir)

    def rotate(self, deg: float, org=None) -> None:
        org = self.translation if org is None else org
        rad = deg2rad(deg)
        self.rotation.yaw += rad
        self.translation = self._rotate_point(self.translation, org, rad)

    def _rotate_point(self, p, org=(0, 0), rad=0):
        mat = [[cos(rad), cos(rad + pi / 2)], [sin(rad), sin(rad + pi / 2)]]
        return (p - org).lmul22(mat) + org
