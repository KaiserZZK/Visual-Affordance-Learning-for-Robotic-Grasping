import os
import pathlib
import argparse
from itertools import chain

import torch
import numpy as np
from skimage.io import imsave

from env import UR5PickEnviornment
from common import load_chkpt, get_splits
import affordance_model

import action_regression_model

import json
from collections import defaultdict
from image import write_rgb
from common import draw_grasp, get_splits
from env import UR5PickEnviornment
from collections import deque 

def main():
    parser = argparse.ArgumentParser(description='Model eval script')
    parser.add_argument('-m', '--model', default="affordance",
        help='which model to evaluate: e.g. "affordance" or "action_regression"')
    parser.add_argument('-t', '--task', default='pick_training',
        help='which task to do: "pick_training" or "empty_bin"')
    parser.add_argument('--headless', action='store_true',
        help='launch pybullet GUI or not')
    parser.add_argument('--seed', type=int, default=10000000,
        help='random seed for empty_bin task')
    parser.add_argument('--save_data', default=False, action='store_true',
        help='save data for additional self-improvement on the affordance model')
    parser.add_argument('--n_past_actions', default=0, type=int)
    args = parser.parse_args()

    if args.model == 'action_regression':
        model_class = action_regression_model.ActionRegressionModel
    else:
        model_class = affordance_model.AffordanceModel

    model_dir = os.path.join('data', args.model)
    chkpt_path = os.path.join(model_dir, 'best.ckpt')

    
    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(n_past_actions=args.n_past_actions)
    model.to(device)
    load_chkpt(model, chkpt_path, device)
    model.eval()

    # load env
    env = UR5PickEnviornment(gui=not args.headless)
    
    if args.task == 'pick_training':
        names = get_splits()['train']
        n_attempts = 3
        vis_dir = os.path.join(model_dir, 'eval_pick_training_vis')
        pathlib.Path(vis_dir).mkdir(parents=True, exist_ok=True)

        results = list()
        for name_idx, name in enumerate(names):
            print('Picking: {}'.format(name))
            env.remove_objects()
            for i in range(n_attempts):
                print('Attempt: {}'.format(i))
                seed = name_idx * 100 + i + 10000
                if i == 0:
                    env.load_ycb_objects([name], seed=seed)
                else:
                    env.reset_objects(seed)
                
                rgb_obs, depth_obs, _ = env.observe()
                coord, angle, vis_img = model.predict_grasp(rgb_obs)
                pick_pose = env.image_pose_to_pick_pose(coord, angle, depth_obs)
                result = env.execute_grasp(*pick_pose)
                print('Success!' if result else 'Failed:(')
                fname = os.path.join(vis_dir, '{}_{}.png'.format(name, i))
                imsave(fname, vis_img)
                results.append(result)
        success_rate = np.array(results, dtype=np.float32).mean()
        print("Testing on training objects. Success rate: {}".format(success_rate))

    elif args.task == 'pick_testing':
        names = get_splits()['test'] 
        n_attempts = 10
        vis_dir = os.path.join(model_dir, 'eval_pick_testing_vis')
        pathlib.Path(vis_dir).mkdir(parents=True, exist_ok=True)
        results = list()
        if args.save_data:
            out_dir = pathlib.Path('eval_data','labels')
            out_dir.mkdir(parents=True, exist_ok=True)
            label_file = out_dir.joinpath('labels.json')
            labels = defaultdict(list)
            # if label_file.is_file():
            #     labels.update(json.load(label_file.open('r'))) 
        for name_idx, name in enumerate(names):
            print('Picking: {}'.format(name))
            env.remove_objects()
            for i in range(n_attempts):
                print('Attempt: {}'.format(i))
                seed = name_idx * 100 + i + 10000
                if i == 0:
                    env.load_ycb_objects([name], seed=seed)
                else:
                    env.reset_objects(seed)
                model.past_actions = deque(maxlen=args.n_past_actions)
                for j in range(args.n_past_actions):
                    rgb_obs, depth_obs, _ = env.observe()
                    coord, angle, vis_img = model.predict_grasp(rgb_obs)
                    # NOTE: check if coord is out of original shape bound
                    img_h, img_w = rgb_obs.shape[:2]
                    coord = list(coord)
                    coord[0] = max(coord[0], 0)
                    coord[0] = min(coord[0], img_w-1)
                    coord[1] = max(coord[1], 0)
                    coord[1] = min(coord[1], img_h-1)
                    coord = tuple(coord)

                    pick_pose = env.image_pose_to_pick_pose(coord, angle, depth_obs)
                    result = env.execute_grasp(*pick_pose)
                    print('Success!' if result else 'Failed:(')
                    fname = os.path.join(vis_dir, '{}_{}.png'.format(name, i))
                    imsave(fname, vis_img)
                    
                    if args.save_data:
                        write_rgb(rgb_obs, str(out_dir.joinpath('{}_{}_rgb.png'.format(name, i))))
                        success = 1 if result else 0
                        labels[name].append([coord[0], coord[1], angle, success])
                        json.dump(labels, label_file.open('w'), indent=4)
                    if result:
                        break
                results.append(result)
                    
        success_rate = np.array(results, dtype=np.float32).mean()
        print("Testing on novel objects. Success rate: {}".format(success_rate))
    else:
        names = list(chain(*get_splits().values()))
        n_attempts = 25
        vis_dir = os.path.join(model_dir, 'eval_empty_bin_vis')
        pathlib.Path(vis_dir).mkdir(parents=True, exist_ok=True)

        print("Loading objects.")
        env.remove_objects()
        env.load_ycb_objects(names, seed=args.seed)
        n_objects = len(names)
        num_in = env.num_object_in_tote1()
        print("{}/{} objects moved".format(n_objects - num_in, n_objects))

        for attempt_id in range(n_attempts):
            print("Attempt {}".format(attempt_id))
            rgb_obs, depth_obs, _ = env.observe()
            coord, angle, vis_img = model.predict_grasp(rgb_obs)
            pick_pose = env.image_pose_to_pick_pose(coord, angle, depth_obs)
            result = env.execute_grasp(*pick_pose)
            if result:
                # place
                env.execute_place()
            
            num_in = env.num_object_in_tote1()
            print("{}/{} objects moved".format(n_objects - num_in, n_objects))

            fname = os.path.join(vis_dir, '{}.png'.format(attempt_id))
            imsave(fname, vis_img)
        print("{} objects left in the bin.".format(num_in))


if __name__ == '__main__':
    main()
