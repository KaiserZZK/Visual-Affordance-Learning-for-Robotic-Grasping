from typing import Tuple, Optional, Dict

import numpy as np
from matplotlib import cm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

from common import draw_grasp
from collections import deque

def get_gaussian_scoremap(
        shape: Tuple[int, int], 
        keypoint: np.ndarray, 
        sigma: float=1, dtype=np.float32) -> np.ndarray:
    """
    Generate a image of shape=:shape:, generate a Gaussian distribtuion
    centered at :keypont: with standard deviation :sigma: pixels.
    keypoint: shape=(2,)
    """
    coord_img = np.moveaxis(np.indices(shape),0,-1).astype(dtype)
    sqrt_dist_img = np.square(np.linalg.norm(
        coord_img - keypoint[::-1].astype(dtype), axis=-1))
    scoremap = np.exp(-0.5/np.square(sigma)*sqrt_dist_img)
    return scoremap

class AffordanceDataset(Dataset):
    """
    Transformational dataset.
    raw_dataset is of type train.RGBDataset
    """
    def __init__(self, raw_dataset: Dataset):
        super().__init__()
        self.raw_dataset = raw_dataset
    
    def __len__(self) -> int:
        return len(self.raw_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Transform the raw RGB dataset element into
        training targets for AffordanceModel.
        return: 
        {
            'input': torch.Tensor (3,H,W), torch.float32, range [0,1]
            'target': torch.Tensor (1,H,W), torch.float32, range [0,1]
        }
        Note: self.raw_dataset[idx]['rgb'] is torch.Tensor (H,W,3) torch.uint8
        """
        # checkout train.RGBDataset for the content of data
        data = self.raw_dataset[idx]
        # TODO: complete this method
        # Hint: Use get_gaussian_scoremap
        # Hint: https://imgaug.readthedocs.io/en/latest/source/examples_keypoints.html
        # ===============================================================================
        norm_angle = float(data['angle']) % 180
        binned_angles = np.arange(8) * (180/8)
        nn_idx = np.argmin(np.abs(binned_angles - norm_angle))
        nn_angle = binned_angles[nn_idx]

        rgb = data['rgb'].numpy().astype(np.float32)/255
        kp = data['center_point']

        kps = KeypointsOnImage([
            Keypoint(x=kp[0], y=kp[1]),
        ], shape=rgb.shape)
        rot_img, rot_kps = iaa.Rotate(-nn_angle)(image=rgb, keypoints=kps)
        goal_kp = rot_kps[0].xy

        goal_img = get_gaussian_scoremap(rgb.shape[:2], goal_kp)

        data = {
            'input': torch.from_numpy(np.moveaxis(rot_img, -1, 0)),
            'target': torch.from_numpy(np.expand_dims(goal_img, 0)), 
        }
        return data
        # ===============================================================================


class AffordanceModel(nn.Module):
    def __init__(self, n_channels: int=3, n_classes: int=1, n_past_actions: int=0, **kwargs):
        """
        A simplified U-Net with twice of down/up sampling and single convolution.
        ref: https://arxiv.org/abs/1505.04597, https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
        :param n_channels (int): number of channels (for grayscale 1, for rgb 3)
        :param n_classes (int): number of segmentation classes (num objects + 1 for background)
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = nn.Sequential(
            # For simplicity:
            #     use padding 1 for 3*3 conv to keep the same Width and Height and ease concatenation
            nn.Conv2d(in_channels=n_channels, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential( 
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.outc = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1)
        # hack to get model device
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.past_actions = deque(maxlen=n_past_actions)

    @property
    def device(self) -> torch.device:
        return self.dummy_param.device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_inc = self.inc(x)  # N * 64 * H * W
        x_down1 = self.down1(x_inc)  # N * 128 * H/2 * W/2
        x_down2 = self.down2(x_down1)  # N * 256 * H/4 * W/4
        x_up1 = self.upconv1(x_down2)  # N * 128 * H/2 * W/2
        x_up1 = torch.cat([x_up1, x_down1], dim=1)  # N * 256 * H/2 * W/2
        x_up1 = self.conv1(x_up1)  # N * 128 * H/2 * W/2
        x_up2 = self.upconv2(x_up1)  # N * 64 * H * W
        x_up2 = torch.cat([x_up2, x_inc], dim=1)  # N * 128 * H * W
        x_up2 = self.conv2(x_up2)  # N * 64 * H * W
        x_outc = self.outc(x_up2)  # N * n_classes * H * W
        return x_outc

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict affordance using this model.
        This is required due to BCEWithLogitsLoss
        """
        return torch.sigmoid(self.forward(x))

    @staticmethod
    def get_criterion() -> torch.nn.Module:
        """
        Return the Loss object needed for training.
        Hint: why does nn.BCELoss does not work well?
        """
        return nn.BCEWithLogitsLoss()

    @staticmethod
    def visualize(input: np.ndarray, output: np.ndarray, 
            target: Optional[np.ndarray]=None) -> np.ndarray:
        """
        Visualize rgb input and affordance as a single rgb image.
        """
        cmap = cm.get_cmap('viridis')
        in_img = np.moveaxis(input, 0, -1)
        pred_img = cmap(output[0])[...,:3]
        row = [in_img, pred_img]
        if target is not None:
            gt_img = cmap(target[0])[...,:3]
            row.append(gt_img)
        img = (np.concatenate(row, axis=1)*255).astype(np.uint8)
        return img


    def predict_grasp(
        self, 
        rgb_obs: np.ndarray,  
    ) -> Tuple[Tuple[int, int], float, np.ndarray]:
        """
        Given an RGB image observation, predict the grasping location and angle in image space.
        return coord, angle, vis_img
        :coord: tuple(int x, int y). By OpenCV convension, x is left-to-right and y is top-to-bottom.
        :angle: float. By OpenCV convension, angle is clockwise rotation.
        :vis_img: np.ndarray(shape=(H,W,3), dtype=np.uint8). Visualize prediction as a RGB image.

        Note: torchvision's rotation is counter clockwise, while imgaug,OpenCV's rotation are clockwise.
        """
        device = self.device
        # TODO: complete this method (prediction)
        # Hint: why do we provide the model's device here?
        # ===============================================================================
        binned_angles = np.arange(8) * (180/8)
        rotators = [iaa.Rotate(-angle) for angle in binned_angles]
        obs_rotated = np.stack([x(image=rgb_obs) for x in rotators])
        rgb = np.moveaxis(obs_rotated,-1,1)

        with torch.no_grad():
            input_batch = torch.from_numpy(rgb).to(
                device=device, dtype=torch.float32) / 255
            affordance_map = self.predict(input_batch).squeeze()
            
            # TODO: (problem 3) avoid selecting the same failed actions
            for max_coord in list(self.past_actions):  
                bin = max_coord[0]
                # ===============================================================================
                keypoint = np.array(max_coord)[1:][::-1]
                supression_map = get_gaussian_scoremap(
                    shape=affordance_map.shape[-2:],
                    keypoint=keypoint,
                    sigma=4)
                affordance_map[bin] -= torch.from_numpy(supression_map)
            
            affordance_map = torch.clip(affordance_map,0,1) 
            affordance_flat = affordance_map.flatten()
        
        max_val, max_idx = affordance_flat.max(dim=0)
        max_val = max_val.detach().to('cpu').numpy()
        max_idx = max_idx.detach().to('cpu').numpy()
        max_coord = np.unravel_index(max_idx, shape=affordance_map.shape)
        self.past_actions.append(max_coord)
        angle = binned_angles[max_coord[0]]
        raw_kps = KeypointsOnImage([
            Keypoint(*max_coord[1:][::-1])],
            shape=rgb_obs.shape)
        rot_kps = iaa.Rotate(angle)(keypoints=raw_kps)
        coord = tuple(rot_kps[0].xy.astype(np.int64).tolist())

        # TODO: complete this method (visualization)
        # :vis_img: np.ndarray(shape=(H,W,3), dtype=np.uint8). Visualize prediction as a RGB image.
        # Hint: use common.draw_grasp
        # Hint: see self.visualize
        # Hint: draw a grey (127,127,127) line on the bottom row of each image.
        # ===============================================================================
        rgb_rotated = input_batch.detach().to('cpu').numpy()
        affordance = affordance_map.detach().to('cpu').numpy()
        rgbs = (np.moveaxis(rgb_rotated,1,-1) * 255).astype(np.uint8)
        cmap = cm.get_cmap('viridis')
        preds = [(cmap(x)[...,:3] * 255).astype(np.uint8) for x in affordance]
        
        img_pairs = list()
        for i, rgb, pred in zip(range(len(rgbs)), rgbs, preds):
            if i == max_coord[0]:
                rgb = rgb.copy()
                draw_grasp(rgb, max_coord[1:][::-1], 0)
            img_pairs.append(np.concatenate([rgb, pred], axis=1))
        
        # arrange
        rows = list()
        for i in range(0,len(rgbs),2):
            x = np.concatenate([img_pairs[i],img_pairs[i+1]], axis=1)
            x[-1,:,:] = 127
            rows.append(x)
        vis_img = np.concatenate(rows, axis=0)
        # ===============================================================================
        return coord, angle, vis_img

