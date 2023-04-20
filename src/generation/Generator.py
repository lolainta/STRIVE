from copy import deepcopy
from NuscData import NuscData
from Dataset import Dataset
from Datalist import Datalist
from Data import Data
from quintic import quintic_polynomials_planner


class Generator:
    def __init__(self, nuscData: NuscData) -> None:
        self.nuscData = nuscData

    def lc(self, d: Data) -> Data:
        ret = deepcopy(d)
        ret.flip()
        ret.forward(ret.length / 2)
        ret.rotate(20, org=d.bound[0])
        return ret

    def gen_by_inst(self, inst: dict) -> Dataset:
        anns = self.nuscData.get_annotations(inst)

        dataset: Dataset = Dataset(self.nuscData.scene, inst)
        ego_data: Datalist = self.nuscData.get_ego_data()
        npc_data: Datalist = self.nuscData.get_npc_data(anns)

        dataset.set_ego(ego_data)
        dataset.set_npc(npc_data)

        ego_final: Data = dataset.ego[-1]
        atk_final: Data = self.lc(ego_final)

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

    def gen_all(self) -> list:
        ret = list()
        inst_tks: set = self.nuscData.instances
        for inst_tk in inst_tks:
            inst = self.nuscData.get("instance", inst_tk)
            ds = self.gen_by_inst(inst)
            ret.append(ds)
        return ret

    def filter(self, dataCluster: list) -> list:
        return [ds for ds in dataCluster if ds.filter()]
