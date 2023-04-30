from copy import deepcopy
from generation.NuscData import NuscData
from generation.Dataset import ColDataset
from generation.Datalist import Datalist
from generation.Data import Data
from generation.quintic import quintic_polynomials_planner
from nuscenes.map_expansion.map_api import NuScenesMap
from enum import Enum, auto
from numpy import rad2deg


class Condition(Enum):
    LC = auto()
    LTAP = auto()
    JC = auto()
    RE = auto()
    HO = auto()


class Generator:
    def __init__(self, nuscData: NuscData) -> None:
        self.nuscData = nuscData

    def LC(self, d: Data) -> Data:
        ret = deepcopy(d)
        ret.flip()
        ret.move(ret.length / 2, 0)
        ret.rotate(20, org=d.bound[0])
        return ret

    def RSide(self, d: Data) -> Data:
        ret = deepcopy(d)
        ret.rotate(90, org=d.bound[0])
        ret.move(ret.length / 2, 90)
        return ret

    def RearEnd(self, d: Data) -> Data:
        ret = deepcopy(d)
        ret.move(ret.length, 180)
        return ret

    def HeadOn(self, d: Data) -> Data:
        ret = deepcopy(d)
        ret.move(ret.length, 180)
        ret.rotate(180)
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
            func = self.RSide
        elif type == Condition.JC:
            func = self.RSide
        elif type == Condition.RE:
            func = self.RearEnd
        elif type == Condition.HO:
            func = self.HeadOn
        else:
            assert False, "Undefined Condition"

        atk_final: Data = func(ego_final)

        nyaw = dataset.npc[0].transform.rotation
        eyaw = dataset.ego[0].transform.rotation
        diff = rad2deg(abs(nyaw - eyaw))
        diff = min(diff, 360 - diff)
        if type in [Condition.LTAP, Condition.JC]:
            if type == Condition.LTAP:
                if not 135 < diff:
                    return None
            elif type == Condition.JC:
                if not (45 < diff and diff < 135):
                    return None
            else:
                return None
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
            res = self.gen_by_inst(inst, type)
            if res is not None:
                ret.append(res)
        return ret

    def filter_by_vel_acc(self, dataCluster: list) -> list:
        return [ds for ds in dataCluster if ds.filter_by_vel_acc()]

    def filter_by_map(self, dataCluster: list, nuscMap: NuScenesMap) -> list:
        return [ds for ds in dataCluster if ds.filter_by_map(nuscMap)]
