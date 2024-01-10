import pandas as pd
import numpy as np
import argparse
import glob
from shapely.geometry import Polygon
from tqdm import tqdm

VEH_COLL_THRESH = (
    0.0  # IoU must be over this to count as a collision for metric (not loss)
)


def get_bbox(row, l, w):
    """
    Get bounding box of vehicle
    :param row: row of dataframe
    :param l: length of vehicle
    :param w: width of vehicle
    :return: bounding box of vehicle (shapely.geometry.Polygon)
    """
    pos = np.array([row["X"], row["Y"]])
    yaw = row["YAW"]  # degree
    yaw = np.deg2rad(yaw)
    u = np.array([np.cos(yaw), np.sin(yaw)]) * l / 2
    v = np.array([np.sin(yaw), -np.cos(yaw)]) * w / 2
    bbox = np.array([pos + u - v, pos - u - v, pos - u + v, pos + u + v])
    return Polygon(bbox)


def check_similarity(ego_yaw, agent_yaw, coll_type) -> bool:
    """
    Check the similarity between desired coll_type and actual coll_type
    :param ego_yaw: ego vehicle yaw(degree)
    :param agent_yaw: agent vehicle yaw(degree)
    :param coll_type: desired coll_type
    :return: True or False
    """
    yaw_offset = 30
    abs_angle = abs(ego_yaw - agent_yaw)
    print("ego_yaw: ", ego_yaw)
    print("agent_yaw: ", agent_yaw)
    print("abs_angle: ", abs_angle)
    if coll_type == "lane_change":
        if abs(abs_angle - 15) < yaw_offset:
            return True
    elif coll_type == "LTAP" or coll_type == "junction_crossing":
        if abs(abs_angle - 90) < yaw_offset or abs(abs_angle - 270) < yaw_offset:
            return True
    elif coll_type == "opposite_direction":
        if abs(abs_angle - 180) < yaw_offset:
            return True
    elif coll_type == "rear_end":
        if abs_angle < yaw_offset:
            return True
    return False


def metric(args):
    vehicle_length = 4.7
    vehicle_width = 2
    folder = args.data_path
    rec_folder = parse_scenario_log_dir(folder)
    print(rec_folder)
    all_cnt = {
        "junction_crossing": 0,
        "LTAP": 0,
        "lane_change": 0,
        "opposite_direction": 0,
        "rear_end": 0,
    }
    coll_cnt = all_cnt.copy()
    simi_cnt = all_cnt.copy()
    for log in tqdm(rec_folder, total=len(rec_folder)):
        flag = False
        coll_type = log.split("/")[-2]
        all_cnt[coll_type] += 1
        df = parse_csv_file(log)
        df = df.groupby("TIMESTAMP")
        for name, group in df:
            for idx, row in group.iterrows():
                if row["OBJECT_TYPE"] == "EGO":
                    ego_bbox = get_bbox(row, vehicle_length, vehicle_width)
                    ego_yaw = row["YAW"]  # degree
                    ego_yaw %= 360
                else:
                    agent_bbox = get_bbox(row, vehicle_length, vehicle_width)
                    agent_yaw = row["YAW"]  # degree
                    agent_yaw %= 360
                    iou = (
                        ego_bbox.intersection(agent_bbox).area
                        / ego_bbox.union(agent_bbox).area
                    )
                    if iou > VEH_COLL_THRESH:
                        print(log)
                        coll_cnt[coll_type] += 1
                        if check_similarity(ego_yaw, agent_yaw, coll_type):
                            simi_cnt[coll_type] += 1
                        flag = True
                        break
            if flag:
                break

    ### print result
    print("all_cnt: ", all_cnt)
    # print collison rate of each type
    print("collision cnt:")
    print("\tjunction crossing: ", coll_cnt["junction_crossing"])
    print("\tLTAP: ", coll_cnt["LTAP"])
    print("\tlane change: ", coll_cnt["lane_change"])
    print("\topposite direction: ", coll_cnt["opposite_direction"])
    print("\trear end: ", coll_cnt["rear_end"])

    print("coll_rate: ")
    print(
        "\tjunction crossing: ",
        coll_cnt["junction_crossing"] / all_cnt["junction_crossing"],
    )
    print("\tLTAP: ", coll_cnt["LTAP"] / all_cnt["LTAP"])
    print("\tlane change: ", coll_cnt["lane_change"] / all_cnt["lane_change"])
    print(
        "\topposite direction: ",
        coll_cnt["opposite_direction"] / all_cnt["opposite_direction"],
    )
    print("\trear end: ", coll_cnt["rear_end"] / all_cnt["rear_end"])
    # print similarity rate of each type
    print("similarity cnt:")
    print("\tjunction crossing: ", simi_cnt["junction_crossing"])
    print("\tLTAP: ", simi_cnt["LTAP"])
    print("\tlane change: ", simi_cnt["lane_change"])
    print("\topposite direction: ", simi_cnt["opposite_direction"])
    print("\trear end: ", simi_cnt["rear_end"])
    print("similarity rate:")
    print(
        "\tjunction crossing: ",
        simi_cnt["junction_crossing"] / coll_cnt["junction_crossing"],
    )
    print("\tLTAP: ", simi_cnt["LTAP"] / coll_cnt["LTAP"])
    print("\tlane change: ", simi_cnt["lane_change"] / coll_cnt["lane_change"])
    print(
        "\topposite direction: ",
        simi_cnt["opposite_direction"] / coll_cnt["opposite_direction"],
    )
    print("\trear end: ", simi_cnt["rear_end"] / coll_cnt["rear_end"])

    # print total information
    print("total_collision rate: ", sum(coll_cnt.values()) / sum(all_cnt.values()))
    print("total_similarity rate: ", sum(simi_cnt.values()) / sum(coll_cnt.values()))


def parse_csv_file(records_file):
    """
    read csv file and return a list of dicts
    :param records_file: path to csv file
    :return: list of dicts
    """

    return pd.read_csv(records_file)


def parse_scenario_log_dir(path):
    """
    Parse the scenario log directory.
    """
    route_scenarios = sorted(
        glob.glob(path + "RouteScenario*/", recursive=True),
        key=lambda x: (x.split("_")[-6]),
    )
    records_files = []
    for dir in route_scenarios:
        records_files.extend(sorted(glob.glob(dir + "**/result_traj.csv")))
    return records_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--future_length", default="12", type=int)
    parser.add_argument("--past_length", default="8", type=int)
    parser.add_argument(
        "--data_path", type=str, default="./out/test_traffic_out_carla_cond_new/result/"
    )
    parser.add_argument("--plot", default=1, type=int)
    args = parser.parse_args()
    metric(args)
