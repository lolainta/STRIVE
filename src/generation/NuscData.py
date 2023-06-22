from nuscenes.nuscenes import NuScenes
from generation.Data import Data
from generation.Datalist import Datalist
from generation.Transform import Transform


class NuscData:
    def __init__(self, nusc: NuScenes, scene: int) -> None:
        self.nusc = nusc
        self.scene_id = scene
        self.scene = self.nusc.scene[scene]
        self.samples: list = self.get_samples()
        self.instances: list = self.get_instances()
        self.times = self.get_time()

    def get(self, table: str, token: str) -> dict:
        return self.nusc.get(table, token)

    def get_map(self) -> str:
        log = self.nusc.get("log", self.scene["log_token"])
        return log["location"]

    def get_samples(self) -> list:
        samples = [self.nusc.get("sample", self.scene["first_sample_token"])]
        while samples[-1]["next"]:
            nxt = self.nusc.get("sample", samples[-1]["next"])
            samples.append(nxt)
        return samples

    def get_instances(self) -> list:
        ret = set()
        for inst in self.nusc.instance:
            ann = self.nusc.get("sample_annotation", inst["first_annotation_token"])

            sample = self.nusc.get("sample", ann["sample_token"])
            if sample["scene_token"] == self.scene["token"]:
                if self.check(inst):
                    ret.add(inst["token"])
        return list(ret)

    def check(self, inst: dict) -> bool:
        if inst["first_annotation_token"] == inst["last_annotation_token"]:
            return False
        cat = self.get("category", inst["category_token"])
        if cat["name"] != "vehicle.car":
            return False
        return True

    def get_annotations(self, inst: dict) -> list:
        fst_tk: str = inst["first_annotation_token"]
        lst_tk: str = inst["last_annotation_token"]
        cur_tk: str = fst_tk
        ann_tks: list[str] = [cur_tk]
        while cur_tk != lst_tk:
            cur = self.nusc.get("sample_annotation", cur_tk)
            cur_tk = cur["next"]
            ann_tks.append(cur_tk)
        anns = [self.nusc.get("sample_annotation", ann_tk) for ann_tk in ann_tks]
        return anns

    def get_npc_data(self, anns: list) -> Datalist:
        ret: Datalist = Datalist()
        for ann in anns:
            sample_tk = ann["sample_token"]
            sample = self.nusc.get("sample", sample_tk)
            ret.append(
                Data(
                    sample["timestamp"], Transform(ann["translation"], ann["rotation"])
                )
            )
        return ret

    def get_ego_data(self) -> Datalist:
        ret: Datalist = Datalist()
        for sample in self.samples:
            data_tk: str = sample["data"]["RADAR_FRONT"]
            data: dict = self.nusc.get("sample_data", data_tk)
            ego_pos_tk: str = data["ego_pose_token"]
            ego_pos: dict = self.nusc.get("ego_pose", ego_pos_tk)
            ret.append(
                Data(
                    sample["timestamp"],
                    Transform(ego_pos["translation"], ego_pos["rotation"]),
                )
            )
        return ret

    def get_time(self) -> list:
        ret = list()
        for sample in self.samples:
            ret.append(sample["timestamp"])
        return ret
