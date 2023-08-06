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


class Generator:
    def __init__(self, nuscData: NuscData) -> None:
        self.nuscData = nuscData
        self.verbose = False

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

    def gen_by_inst(self, inst_anns: list, ego_data: Datalist) -> list:
        npc_data: Datalist = self.nuscData.get_npc_data(inst_anns)
        npc_data.compile()
        ops = {
            self.LLC: 1,
            self.RLC: 1,
            self.LSide: 12,
            self.RSide: 12,
            self.RearEnd: 2,
            self.HeadOn: 1,
        }

        ret = list()
        for tid, last_frame in enumerate(self.nuscData.times):
            if last_frame - self.nuscData.times[0] < 8000000:
                continue
            ego_final: Data = ego_data[tid]
            for fid, op in enumerate(ops.items()):
                func, num = op
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
                        if len(npc_data) < 5:
                            continue
                        eyaw = rad2deg(ego_data[-5].transform.rotation.yaw)
                        nyaw = rad2deg(npc_data[-5].transform.rotation.yaw)
                        diff = abs(eyaw - nyaw)
                        diff = min(diff, 360 - diff)
                        if diff > 150:
                            cond = Condition.LTAP
                        elif 60 < diff < 120:
                            cond = Condition.JC
                        else:
                            cond = None
                    if cond is not None:
                        ret.append((f"{fid}-{idx}-{tid}", res, cond))
        return ret

    def gen_all(self) -> list:
        ret = list()
        inst_tks: list = self.nuscData.instances
        inst_anns: list = [
            self.nuscData.get_annotations(self.nuscData.get("instance", tk))
            for tk in inst_tks
        ]
        assert len(inst_tks) == len(inst_anns)
        for anns in inst_anns:
            ego_data: Datalist = self.nuscData.get_ego_data()
            ego_data.compile()
            res = self.gen_by_inst(anns, ego_data)
            for idx, r, cond in res:
                new_ego = deepcopy(ego_data)
                col = ColDataset(
                    self.nuscData.scene,
                    self.nuscData.get("instance", anns[0]["instance_token"]),
                    cond,
                )
                col.set_ego(new_ego)
                col.set_atk(r)
                for npcs in inst_anns:
                    if npcs is not anns:
                        col.add_npc(
                            self.nuscData.get_npc_data(npcs), npcs[0]["instance_token"]
                        )
                col.idx = idx
                timelist = col.atk.gen_timelist()
                col.trim(timelist)
                col.ego.gen_timelist()
                assert col.atk.timelist == col.ego.timelist
                ret.append(col)
        if self.verbose:
            print(f"scene[{self.nuscData.scene_id}] Generated {len(ret)} data")
        return ret

    def filter_by_vel_acc(self, dataCluster: list) -> list:
        ret = [ds for ds in dataCluster if ds.filter_by_vel_acc()]
        if self.verbose:
            print(
                f"scene[{self.nuscData.scene_id}] Filtered {len(dataCluster) - len(ret)} data by vel and acc"
            )
        return ret

    def filter_by_map(self, dataCluster: list, nuscMap: NuScenesMap) -> list:
        ret = [ds for ds in dataCluster if ds.filter_by_map(nuscMap)]
        if self.verbose:
            print(
                f"scene[{self.nuscData.scene_id}] Filtered {len(dataCluster) - len(ret)} data by map"
            )
        return ret

    def filter_by_collision(self, dataCluster: list) -> list:
        ret = [ds for ds in dataCluster if ds.filter_by_collision()]
        if self.verbose:
            print(
                f"scene[{self.nuscData.scene_id}] Filtered {len(dataCluster) - len(ret)} data by collision"
            )
        return ret

    def filter_by_curvature(self, dataCluster: list) -> list:
        ret = [ds for ds in dataCluster if ds.filter_by_curvature()]
        if self.verbose:
            print(
                f"scene[{self.nuscData.scene_id}] Filtered {len(dataCluster) - len(ret)} data by curvature"
            )
        return ret
