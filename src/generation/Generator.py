from copy import deepcopy
from generation.NuscData import NuscData
from generation.Dataset import ColDataset
from generation.Datalist import Datalist
from generation.Data import Data
from generation.Condition import Condition
from generation.quintic import quintic_polynomials_planner
from nuscenes.map_expansion.map_api import NuScenesMap
from numpy import rad2deg
from random import randint
from copy import deepcopy
from icecream import ic


class Generator:
    def __init__(self, nuscData: NuscData) -> None:
        self.nuscData = nuscData

    def fetch_data(self):
        self.nuscData.fetch_data()

    def LLC(self, d: Data) -> Data:
        ret = deepcopy(d)
        ret.move(ret.width, 90)
        ret.move(ret.length / 2, 0)
        ret.rotate(-20, org=d.bound[1])
        return ret

    def RLC(self, d: Data) -> Data:
        ret = deepcopy(d)
        ret.move(ret.width, -90)
        ret.move(ret.length / 2, 0)
        ret.rotate(20, org=d.bound[0])
        return ret

    def LSide(self, d: Data) -> Data:
        ret = deepcopy(d)
        ret.rotate(-90, org=d.bound[1])
        ret.move(ret.length / 2 - ret.width / 2, -90)
        return ret

    def RSide(self, d: Data) -> Data:
        ret = deepcopy(d)
        ret.rotate(90, org=d.bound[0])
        ret.move(ret.length / 2 - ret.width / 2, 90)
        return ret

    def RearEnd(self, d: Data) -> Data:
        ret = deepcopy(d)
        ret.move(ret.length, 180)
        return ret

    def HeadOn(self, d: Data) -> Data:
        ret = deepcopy(d)
        ret.move(ret.length, 0)
        ret.rotate(180)
        return ret

    def encapsulate_collision(
        self,
        idx: str,
        atk: list,
        cond: Condition,
        inst: dict,
        ego: Datalist,
        inst_anns: list,
    ) -> ColDataset:
        cond = deepcopy(cond)
        ego_data: Datalist = deepcopy(ego)
        col = ColDataset(
            self.nuscData.scene,
            inst,
            cond,
        )
        col.set_ego(ego_data)
        col.set_atk(atk)
        for npcs in inst_anns:
            if npcs is not inst_anns:
                col.add_npc(self.nuscData.get_npc_data(npcs), npcs[0]["instance_token"])
        col.idx = idx
        timelist = col.atk.gen_timelist()
        col.trim(timelist)
        col.ego.gen_timelist()
        assert col.atk.timelist == col.ego.timelist
        return col

    def gen_by_inst(
        self,
        npc_data: Datalist,
        ego_data: Datalist,
        map: NuScenesMap,
    ) -> list:
        def is_intersection(x, y):
            rstk = map.record_on_point(x, y, "road_segment")
            if rstk == "":
                return False
            rs = map.get("road_segment", rstk)
            return rs["is_intersection"]

        npc_data.compile()
        ops = {
            self.LLC: 2,
            self.RLC: 2,
            self.LSide: 12,
            self.RSide: 12,
            self.RearEnd: 8,
            self.HeadOn: 1,
        }
        ret = list()
        for tid in range(10, 17):
            if tid >= len(ego_data):
                break
            ego_final: Data = ego_data[tid]
            for fid, op in enumerate(ops.items()):
                func, num = op

                if func == self.LSide:
                    # check if ego at left hand side
                    mid = ego_data[0].transform.rotation
                    atk_rel = npc_data[0].transform.rotation - mid
                    if atk_rel < 0:
                        continue
                elif func == self.RSide:
                    # check if ego at right hand side
                    mid = ego_data[0].transform.rotation
                    atk_rel = npc_data[0].transform.rotation - mid
                    if atk_rel > 0:
                        continue

                atk_final: Data = func(ego_final)
                for idx in range(num):
                    if func == self.LLC:
                        atk_final.rotate(randint(0, 10), org=ego_final.bound[1])
                    elif func == self.RLC:
                        atk_final.rotate(randint(0, 10), org=ego_final.bound[0])
                    elif func in [self.LSide, self.RSide]:
                        atk_final.move(randint(-5, 5) / 10, 90)
                    elif func == self.RearEnd:
                        atk_final.move(randint(-10, 10) / 10, 90)
                    elif func == self.HeadOn:
                        atk_final.move(randint(-10, 10) / 10, 90)
                    else:
                        assert False, "Unknown function"

                    res = quintic_polynomials_planner(
                        src=npc_data[0].transform,
                        sv=npc_data[0].velocity,
                        sa=npc_data[0].accelerate,
                        dst=atk_final.transform,
                        gv=npc_data[-1].velocity,
                        ga=npc_data[-1].accelerate,
                        timelist=self.nuscData.times[: tid + 1],
                    )
                    res.gen_timelist()
                    assert res.timelist == self.nuscData.times[: tid + 1]

                    if func in [self.LLC, self.RLC]:
                        cond = Condition.LC
                    elif func == self.RearEnd:
                        cond = Condition.RE
                    elif func == self.HeadOn:
                        cond = Condition.HO
                    else:
                        assert len(ego_data[: tid + 1]) == len(
                            res
                        ), f"{tid} {len(npc_data)} {len(res)}"
                        for i in range(1, len(res)):
                            if is_intersection(
                                res[i].transform.translation.x,
                                res[i].transform.translation.y,
                            ):
                                continue
                            diff = (
                                ego_data[tid - i].transform.rotation
                                - res[-i].transform.rotation
                            )
                            diff = abs(rad2deg(diff))
                            eps = 15
                            if abs(diff - 180) < eps:
                                cond = Condition.LTAP
                            elif abs(diff - 90) < eps:
                                cond = Condition.JC
                            else:
                                cond = None
                            break
                        if cond is None:
                            continue

                        # if len(npc_data) < 5:
                        #     continue
                        # diff = (
                        #     ego_data[tid - 5].transform.rotation
                        #     - res[-5].transform.rotation
                        # )
                        # diff = abs(rad2deg(diff))
                        # eps = 15
                        # if abs(diff - 180) < eps:
                        #     cond = Condition.LTAP
                        # elif abs(diff - 90) < eps:
                        #     cond = Condition.JC
                        # else:
                        #     cond = None
                        # if cond is None:
                        #     continue

                    ret.append((f"{fid}-{idx}-{tid}", res, deepcopy(cond)))
        return ret

    def gen_all(self, map: NuScenesMap) -> list:
        ret = list()
        self.nuscData.fetch_data()
        inst_tks: list = self.nuscData.inst_tks
        insts: list = self.nuscData.insts
        inst_anns: list = self.nuscData.inst_anns
        npcs_data: list = self.nuscData.npcs
        assert len(inst_tks) == len(npcs_data)
        for inst, npc_data in zip(insts, npcs_data):
            ego_data: Datalist = self.nuscData.get_ego_data()
            ego_data.compile()
            res = self.gen_by_inst(npc_data, ego_data, map)
            for idx, r, cond in res:
                col = self.encapsulate_collision(
                    idx, r, cond, inst, ego_data, inst_anns
                )
                if col.filter():
                    ret.append(col)
        return ret

    def filter_by_vel_acc(self, dataCluster: list) -> list:
        ret = [ds for ds in dataCluster if ds.filter_by_vel_acc()]
        return ret

    def filter_by_map(self, dataCluster: list, nuscMap: NuScenesMap) -> list:
        ret = [ds for ds in dataCluster if ds.filter_by_map(nuscMap)]
        return ret

    def filter_by_collision(self, dataCluster: list) -> list:
        ret = [ds for ds in dataCluster if ds.filter_by_collision()]
        return ret

    def filter_by_curvature(self, dataCluster: list) -> list:
        ret = [ds for ds in dataCluster if ds.filter_by_curvature()]
        return ret
