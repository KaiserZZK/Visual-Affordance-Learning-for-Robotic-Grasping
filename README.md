# Visual-Affordance-Learning-for-Robotic-Grasping
Visual Affordance model for robotic grasping, inspired by MIT-Princeton's robotic pick-and-place system (https://vision.princeton.edu/projects/2017/arc/), built with PyTorch and the Mini U-Net architecture.

## Setting up virtual environment

1. Install python interpreter and dependencies using [miniforge](https://github.com/conda-forge/miniforge#mambaforge)
(also works for Apple Silicon, comparing to miniconda). 
Please download the corresponding `Mambaforge-OS-arch.sh/exe` file 
and follow the [installation instructions](https://github.com/conda-forge/miniforge#install). 
2. After installation and initialization (i.e. conda init), 
launch a new terminal and run:
```shell
mamba env create -f environment gpu.yaml
```
if you have an Nvidia GPU on your computer, or
```shell
mamba env create -f environment cpu.yaml
```
otherwise, inside the home directory (`./`) . 
This will create a new `venv` environment, 
which can be activated by running 
```shell
mamba activate venv
```

## Generating training data
The training data are geenrated in this direcotry; but if ur intertested
in customizing 

(explain it a bit) pixel x,y, and angle, rotation we follow [the
OpenCV convention for pixel coordinate and rotation angle]
(https://learnopencv.com/image-rotation-and-translation-using-opencv/)

```shell
python3 pick labeler.py
```

the script will launch a PyBullet GUI, then load the robot and objects. 
An OpenCV window will pop up (Fig. 3), where

you can click left mouse button to select grasping location and use A and D 
keys to rotate the grasping orientation. To confirm the grasp pose, press Q 
or Enter.

kindly make sure all angles are positive 

Label 5 training objects with 12 attempts each. This usually take around 
5 minutes. You might notice one of the objects, namely YcbMediumClamp, shows 
up similar to a small dot and makes it hard identify a grasping pose: try to 
click its center and position the gripper in any reasonable orientation.

