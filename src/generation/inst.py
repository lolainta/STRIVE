import numpy as np
from nuscenes.nuscenes import NuScenes
from src.Drawer import Drawer
from quintic import quintic_polynomials_planner
from NuscData import NuscGenerator


def main():
    nusc = NuScenes(version="v1.0-mini", dataroot="./data", verbose=True)

    gen = NuscGenerator(nusc, 5)

    # scene = nusc.scene[5]
    # samples = gen.getSamples(scene)

    # nusc.render_scene(scene['token'])

    # ann_tk = samples[30]['anns'][7]
    # ann = nusc.get('sample_annotation', ann_tk)
    # print(ann_tk)
    # # nusc.render_annotation(ann_tk)
    # # inst_tk = 69385845cb9747b7afe095177cc405b5
    # inst_tk = ann['instance_token']
    # print(inst_tk)
    # inst = nusc.get('instance', inst_tk)
    # # print(inst)

    # anns = gen.getAnnotations(inst)

    # # print(ann)
    # # print([sample['token'] for sample in samples],anns)
    # # print(len(samples),len(anns))
    # # for sample, ann in zip(samples, anns):
    # #     assert ann['token'] in sample['anns']
    # #     assert ann['sample_token'] == sample['token']
    # # print('sample', sample)
    # # print('ann', ann)

    # data = gen.compile_data(samples, anns)
    # # print([d['time'] for d in data])
    # elapse_time = (data[-1]['time']-data[0]['time'])/1000000
    # print(f'{elapse_time=}s')
    # print('Data compiled successfully.')

    # print('Drawing data')
    # plt = Drawer()
    # plt.plot_data(data)
    # print('Data drawn')

    # final = data[-1]
    # ego = np.array([final['ego_x'], final['ego_y']])
    # eyaw = final['ego_yaw']

    # atk = (ego[0]+3*np.cos(eyaw-np.pi/2), ego[1]+3*np.sin(eyaw-np.pi/2))
    # ayaw = eyaw

    # atk = atk+0*np.array([np.cos(ayaw), np.sin(ayaw)])
    # ayaw = ayaw+np.deg2rad(20)

    # plt.plot_car(atk[0], atk[1], ayaw)
    # # plt.show()

    # res = quintic_polynomials_planner(
    #     data[0]['atk_x'], data[0]['atk_y'], data[0]['atk_yaw'], 5, 0.1, atk[0], atk[1], ayaw, 10, -1, 10, 5, 0.5, elapse_time-0.5, elapse_time+0.5)
    # # print(res)
    # plt.show_animation(data, res)

    # # plt.show()


if __name__ == "__main__":
    main()
