from typing import Dict, List, Tuple

import pathlib
import numpy as np
import cv2
import torch
import os
import pandas as pd


def get_splits() -> Dict[str, List[str]]:
    """
    Read read training/testing split.
    """
    csv_path = os.path.join(os.path.dirname(__file__),'assets','split.csv')
    df = pd.read_csv(csv_path, header=0)
    splits = {
        'train': list(df['Name'].loc[df['Split'] == 'Train']),
        'test': list(df['Name'].loc[pd.isnull(df['Split'])])
    }
    return splits


def draw_grasp(
        img: np.ndarray, 
        coord: Tuple[int,int], angle: float, 
        width: int=6, thickness: int=1, 
        radius: int=2, color=(255,255,255)) -> None:
    """
    Draw grasp pose visualization on :img:.
    :img: (H,W,3) RGB image to be MODIFIED
    :coord: tuple(int x, int y). By OpenCV convension, x is left-to-right and y is top-to-bottom.
    :angle: float. By OpenCV convension, angle is clockwise rotation.
    :width: width of gripper in pixel
    :thickness: thickness of line in pixel
    :radius: radius of circles in pixel
    :color: color of symbol in RGB
    """
    center_coord = np.array(coord, dtype=np.float32)
    rad = angle / 180 * np.pi
    direction = np.array([np.cos(rad), np.sin(rad)], dtype=np.float32)
    left_coord = center_coord - direction * width
    right_coord = center_coord + direction * width
    cv2.polylines(img, 
        [np.array([left_coord, right_coord]).round().astype(np.int32)], 
        isClosed=False, 
        color=color, thickness=thickness, lineType=cv2.LINE_AA)
    for point in np.array([left_coord, center_coord, right_coord]).round().astype(np.int32):
        cv2.circle(img, center=point, radius=radius, color=color, 
            thickness=cv2.FILLED, lineType=cv2.LINE_AA)


def save_chkpt(model, epoch, test_loss, chkpt_path):
    """
        Save the trained model.
        :param model (torch.nn.module object): miniUNet object in this homework, trained model.
        :param epoch (int): current epoch number.
        :param test_miou (float): miou of the test set.
        :return: None
    """
    pathlib.Path(chkpt_path).parent.mkdir(parents=True, exist_ok=True)
    state = {'model_state_dict': model.state_dict(),
             'epoch': epoch,
             'model_loss': test_loss, }
    torch.save(state, chkpt_path)
    print("checkpoint saved at epoch", epoch)


def load_chkpt(model, chkpt_path, device):
    """
        Load model parameters from saved checkpoint.
        :param model (torch.nn.module object): miniUNet model to accept the saved parameters.
        :param chkpt_path (str): path of the checkpoint to be loaded.
        :return model (torch.nn.module object): miniUNet model with its parameters loaded from the checkpoint.
        :return epoch (int): epoch at which the checkpoint is saved.
        :return model_miou (float): miou of the test set at the checkpoint.
    """
    checkpoint = torch.load(chkpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    model_loss = checkpoint['model_loss']
    print("epoch, model_loss:", epoch, model_loss)
    return model, epoch, model_loss
