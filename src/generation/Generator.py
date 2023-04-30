from copy import deepcopy
from generation.NuscData import NuscData
from generation.Dataset import ColDataset
from generation.Datalist import Datalist
from generation.Data import Data
from generation.quintic import quintic_polynomials_planner
from nuscenes.map_expansion.map_api import NuScenesMap
from enum import Enum, auto


class Condition(Enum):
    LC = auto()
    LTAP = auto()


class Generator:
    def __init__(self, nuscData: NuscData) -> None:
        self.nuscData = nuscData

    def LC(self, d: Data) -> Data:
        ret = deepcopy(d)
        ret.flip()
        ret.move(ret.length / 2, 0)
        ret.rotate(20, org=d.bound[0])
        return ret

    def LTAP(self, d: Data) -> Data:
        ret = deepcopy(d)
        d.get_bound()
        ret.rotate(90, org=d.bound[0])
        ret.move(ret.length / 2, 90)
        return ret

    def gen_by_inst(self, inst: dict, type: Condition) -> ColDataset:
        anns = self.nuscData.get_annotations(inst)

        dataset: ColDataset = ColDataset(self.nuscData.scene, inst)
        ego_data: Datalist = self.nuscData.get_ego_data()
        npc_data: Datalist = self.nuscData.get_npc_data(anns)

        dataset.set_ego(ego_data)
        dataset.set_npc(npc_data)

        ego_final: Data = dataset.ego[-1]

        func = None
        if type == Condition.LC:
            func = self.LC
        elif type == Condition.LTAP:
            func = self.LTAP
        else:
            assert False, "Undefined Condition"

        atk_final: Data = func(ego_final)

        res = quintic_polynomials_planner(
            src=dataset.npc[0].transform,
            sv=dataset.npc[0].velocity,
            sa=dataset.npc[-1].accelerate,
            dst=atk_final.transform,
            gv=dataset.npc[-1].velocity,
            ga=dataset.npc[-1].accelerate,
            timelist=dataset.timelist,
        )

        dataset.set_atk(res)

        return dataset

    def gen_all(self, type: Condition) -> list:
        ret = list()
        inst_tks: set = self.nuscData.instances
        for inst_tk in inst_tks:
            inst = self.nuscData.get("instance", inst_tk)
            ds = self.gen_by_inst(inst, type)
            ret.append(ds)
        return ret

    def filter_by_vel_acc(self, dataCluster: list) -> list:
        return [ds for ds in dataCluster if ds.filter_by_vel_acc()]

    def filter_by_map(self, dataCluster: list, nuscMap: NuScenesMap) -> list:
        return [ds for ds in dataCluster if ds.filter_by_map(nuscMap)]
