from copy import deepcopy
from generation.NuscData import NuscData
from generation.Dataset import ColDataset
from generation.Datalist import Datalist
from generation.Data import Data
from generation.Condition import Condition
from generation.quintic import quintic_polynomials_planner
from nuscenes.map_expansion.map_api import NuScenesMap
from numpy import rad2deg


class Generator:
    def __init__(self, nuscData: NuscData) -> None:
        self.nuscData = nuscData

    def LC(self, d: Data) -> Data:
        ret = deepcopy(d)
        ret.flip()
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

    def gen_by_inst(self, inst_anns: list, ego_data: Datalist, type: Condition) -> list:
        # dataset: ColDataset = ColDataset(self.nuscData.scene, inst)
        npc_data: Datalist = self.nuscData.get_npc_data(inst_anns)
        npc_data.compile()
        ego_final: Data = ego_data[-1]
        func = []
        if type == Condition.LC:
            func = [self.LC]
        elif type == Condition.TB:
            func = [self.LSide, self.RSide]
        elif type == Condition.RE:
            func = [self.RearEnd]
        elif type == Condition.HO:
            func = [self.HeadOn]
        else:
            assert False, "Undefined Condition"

        ret = list()
        for function in func:
            atk_final: Data = function(ego_final)

            res = quintic_polynomials_planner(
                src=npc_data[0].transform,
                sv=npc_data[0].velocity,
                sa=npc_data[-1].accelerate,
                dst=atk_final.transform,
                gv=npc_data[-1].velocity,
                ga=npc_data[-1].accelerate,
                timelist=self.nuscData.times,
            )
            ret.append(res)
        return ret

    def gen_all(self, cond: Condition) -> list:
        ret = list()
        inst_tks: list = self.nuscData.instances
        inst_anns: list = [
            self.nuscData.get_annotations(self.nuscData.get("instance", tk))
            for tk in inst_tks
        ]
        assert len(inst_tks) == len(inst_anns)
        # print(inst_anns[0][0].keys(), 123)
        for anns in inst_anns:
            ego_data: Datalist = self.nuscData.get_ego_data()
            ego_data.compile()
            res = self.gen_by_inst(anns, ego_data, cond)
            for r in res:
                col = ColDataset(
                    self.nuscData.scene,
                    self.nuscData.get("instance", anns[0]["instance_token"]),
                    cond,
                )
                col.set_ego(ego_data)
                col.set_atk(r)
                for npcs in inst_anns:
                    if npcs is not anns:
                        col.add_npc(self.nuscData.get_npc_data(npcs))
                ret.append(col)
        return ret

    def filter_by_vel_acc(self, dataCluster: list) -> list:
        return [ds for ds in dataCluster if ds.filter_by_vel_acc()]

    def filter_by_map(self, dataCluster: list, nuscMap: NuScenesMap) -> list:
        return [ds for ds in dataCluster if ds.filter_by_map(nuscMap)]
