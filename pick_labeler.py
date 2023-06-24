import pathlib
import json
from collections import defaultdict
from typing import Tuple

import numpy as np
import cv2

from image import write_rgb
from common import draw_grasp, get_splits
from env import UR5PickEnviornment


class GraspLabeler:
    """Function object for GUI labeling"""
    def __init__(self):
        """
        :coord: tuple(int x, int y). By OpenCV convension, x is left-to-right and y is top-to-bottom.
        :angle: float. By OpenCV convension, angle is clockwise rotation.
        """
        self.coord = None
        self.angle = 0
        self.angle_delta = 180 / 8
    
    def callback(self, action, x, y, flags, *userdata):
        """
        https://docs.opencv.org/4.5.5/d7/dfc/group__highgui.html#gab7aed186e151d5222ef97192912127a4
        """
        # TODO: complete this method
        # set gripping point self.coord to the left mouse button clicked point
        # ===============================================================================
        # pass
        if action == cv2.EVENT_LBUTTONDOWN:
            coord = (x,y)
            self.coord = coord
        # ===============================================================================
    
    def __call__(self, img: np.ndarray
            ) -> Tuple[Tuple[int, int], float]:
        """
        Main loop for GUI.
        :img: RGB observation
        :coord: tuple(int x, int y). By OpenCV convension, x is left-to-right and y is top-to-bottom.
        :angle: float. By OpenCV convension, angle is clockwise rotation.
        """
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("img", self.callback)

        vis_img = img.copy()
        while True:
            vis_img[:] = img[:]
            if self.coord is not None:
                draw_grasp(vis_img, self.coord, self.angle)

            cv2.imshow("img", vis_img[:,:,[2,1,0]])
            key = cv2.waitKey(17)
            if key == ord('q'):
                # print('exit')
                break
            elif key == ord('a'):
                # TODO: complete this method
                # rotate gripper counter clockwise by angle_delta
                # by changing self.angle
                # ===============================================================================
                self.angle -= self.angle_delta
            elif key == ord('d'):
                # TODO: complete this method
                # rotate gripper clockwise by angle_delta
                # by changing self.angle
                # ===============================================================================
                self.angle += self.angle_delta
            elif key == 13:
                # print('enter')
                break
        return self.coord, self.angle


def main():
    env = UR5PickEnviornment(gui=True)
    names = get_splits()['train']
    n_picks = 12

    out_dir = pathlib.Path('data','labels')
    out_dir.mkdir(parents=True, exist_ok=True)
    label_file = out_dir.joinpath('labels.json')
    labels = defaultdict(list)
    if label_file.is_file():
        labels.update(json.load(label_file.open('r')))

    for name_idx, name in enumerate(names):
        if len(labels[name]) >= n_picks:
            print("Enough labels for {}".format(name))
            continue
        start_idx = len(labels[name])
        env.load_ycb_objects([name], seed=name_idx*100+start_idx)
        for i in range(start_idx, n_picks):
            while True:
                seed = name_idx*100+i
                env.reset_objects(seed)
                print('Labeling {}_{}'.format(name, i))
                rgb_obs, depth_obs, mask_obs = env.observe()
                # get label
                coord, angle = GraspLabeler()(rgb_obs)
                if coord is None:
                    print("Invalid label, please retry!")
                    continue
                pick_pose = env.image_pose_to_pick_pose(coord, angle, depth_obs)
                result = env.execute_grasp(*pick_pose)
                if result:
                    break
                else:
                    print("Failed to grasp, please retry!")
            # save
            write_rgb(rgb_obs, str(out_dir.joinpath('{}_{}_rgb.png'.format(name, i))))
            labels[name].append([coord[0],coord[1],angle])
            json.dump(labels, label_file.open('w'), indent=4)
        env.remove_objects()


if __name__ == '__main__':
    main()
