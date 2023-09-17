import numpy as np
from generation.Data import Data
from generation.Datalist import Datalist
from generation.Transform import Transform


class QuinticPolynomial:
    def __init__(self, xs, vxs, axs, xe, vxe, axe, time):
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array(
            [
                [time**3, time**4, time**5],
                [3 * time**2, 4 * time**3, 5 * time**4],
                [6 * time, 12 * time**2, 20 * time**3],
            ],
            dtype="float64",
        )
        b = np.array(
            [
                xe - self.a0 - self.a1 * time - self.a2 * time**2,
                vxe - self.a1 - 2 * self.a2 * time,
                axe - 2 * self.a2,
            ],
            dtype="float64",
        )
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):
        xt = (
            self.a0
            + self.a1 * t
            + self.a2 * t**2
            + self.a3 * t**3
            + self.a4 * t**4
            + self.a5 * t**5
        )
        return xt

    def calc_first_derivative(self, t):
        xt = (
            self.a1
            + 2 * self.a2 * t
            + 3 * self.a3 * t**2
            + 4 * self.a4 * t**3
            + 5 * self.a5 * t**4
        )
        return xt

    def calc_second_derivative(self, t):
        xt = (
            2 * self.a2
            + 6 * self.a3 * t
            + 12 * self.a4 * t**2
            + 20 * self.a5 * t**3
        )
        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t**2
        return xt


def quintic_polynomials_planner(
    src: Transform, sv, sa, dst: Transform, gv, ga, timelist: list
) -> Datalist:
    ret = Datalist()
    sx = src.translation.x
    sy = src.translation.y
    syaw = src.rotation.yaw

    gx = dst.translation.x
    gy = dst.translation.y
    gyaw = dst.rotation.yaw

    vxs = sv * np.cos(syaw)
    vys = sv * np.sin(syaw)
    vxg = gv * np.cos(gyaw)
    vyg = gv * np.sin(gyaw)

    axs = sa * np.cos(syaw)
    ays = sa * np.sin(syaw)
    axg = ga * np.cos(gyaw)
    ayg = ga * np.sin(gyaw)

    base_time = timelist[0] / 1e6
    final_time = timelist[-1] / 1e6
    total_time = final_time - base_time

    xqp = QuinticPolynomial(sx, vxs, axs, gx, vxg, axg, total_time)
    yqp = QuinticPolynomial(sy, vys, ays, gy, vyg, ayg, total_time)
    for time in timelist:
        t = time / 1e6 - base_time
        x = xqp.calc_point(t)
        y = yqp.calc_point(t)

        vx = xqp.calc_first_derivative(t)
        vy = yqp.calc_first_derivative(t)
        yaw = np.arctan2(vy, vx)
        ret.append(Data(time, Transform([x, y, 0], yaw)))
    return ret
