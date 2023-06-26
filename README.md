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
mamba env create -f environment_gpu.yaml
```
if you have an Nvidia GPU on your computer, or
```shell
mamba env create -f environment_cpu.yaml
```
otherwise, inside the home directory (`./`) . 
This will create a new `venv` environment, 
which can be activated by running 
```shell
mamba activate venv
```

## Visual Affordance


When designing learning algorithms for robots, how to represent a robot’s observation 
input and action outputs often play a decisive role in the algorithm’s learning 
efficiency and generalization ability. In this project, we explore a specific type 
of action representation, Visual Affordance (also called Spatial Action Map): 
an algorithm for visual robotic pack-and-place tasks.

There are Two key assumptions in this approach:
1. The robot arm’s image observations come from a top-down camera, 
and the entire workspace is visible.
2. The robot performs only top-down grasping, where the pose of 
the gripper is reduced to 3 degrees of freedom (2D translation and 1D rotation).

Under these assumptions, we can easily align actions with image observations 
(hence the name spatial-action map). 
**Visual Affordance is defined as a per-pixel value between 0 and 1 that represents 
whether the pixel (or the action directly mapped to this pixel) is _graspable_.** 
Using this representation, the transnational degrees of freedom are naturally 
encoded in the 2D pixel coordinates. To encode the rotational degree of freedom, 
we rotate the image observation in the opposite direction of gripper rotation 
before passing it to the network, effectively simulating a wrist-mounted camera 
that rotates with the gripper.

## Generating training data
The training data has been generated and could be found in [this direcotry](./data/labels); 
feel free to read this section for more details on the training data generation process. 

You may check the labelled data at [./data/labels/labels.json](./data/labels/labels.json), 
an example entry is given below:
```json
{
    "YcbHammer": [
        [
            75,
            64,
            45.0
        ]
    ]
}
```
There are 5 objects, and 12 data entries for each object; each data entry is formatted as 
`(x, y, angle)`, where `(x, y)` are the pixel coordinates in a 2D image, and `angle` 
represents the robot's gripper orientation. Here, we follow the OpenCV convention for 
pixel coordinate and rotation angle:

![the OpenCV convention for 
pixel coordinate and rotation angle](./illustrations/OpenCV.png).

To replicate the process and generate your own data, execute the following 
command in your terminal:
```shell
python3 pick labeler.py
```

The script will launch a PyBullet GUI, then load the robot and objects. 
An OpenCV window will pop up: 

![OpenCV GUI](./illustrations/GUI.png) 

where you can click left mouse button to 
select grasping location and use A and D keys to rotate the grasping orientation. 
To confirm the grasp pose, press Q or Enter.

We label 5 training objects with 12 attempts each. This usually take around 
5 minutes. You might notice one of the objects, namely YcbMediumClamp, shows 
up similar to a small dot and makes it hard identify a grasping pose: try to 
click its center and position the gripper in any reasonable orientation.


## Model implementation and training 

We implement training related features for the Visual Affordance model using 
the MiniUNet architecture:

![the MiniUNet architecture](./illustrations/MiniUNet.png)

### 2a&b: `RGBDataset` in `train.py` and `AffordanceDataset` in `affordance_model.py`

We use a custom method `get_gaussian_scoremap` to generate the 
affordance target, instead of a one-hot pixel image.

Based on comparison with the absense of gaussian scoremap, we discovered
that using gaussian scoremap to generate the affordance target has a better 
performance than a one-hot pixel image.

Here is the reason: 
gaussian scoremap is able to incorporate the information around the neighbors 
of the center pixel, whereas the one-hot pixel only captures the single value 
information in that pixel. The gaussian scoremap outputs a gaussian distribution of the probability of grasping 
around that a certain pixel, allowing the model to be more robust and generalizable.

### 2c: `AugumentedDataset` in `train.py`

`self.aug pipeline` in the `AugumentedDataset` class applies a transformation with a probability 
of 70%. This transformation is: 
1. translating the image horizontally and vertically by a percentage from (-0.2,0.2);
2. rotating it by an angle from (−δA/2,δA/2), where A is the size of the `bin`, 
which is 22.5 degrees.


### 2d: Training

Execute the following command to start training your model:
```shell
python3 train.py --model affordance --augmentation
```
This script trains the `AffordanceModel` for 101 epochs until convergence.

The train loss and test (validation) loss should be around 0.0012 and 0.0011 respectively.

Here is an example visualization of the training process (from left to right: input, perdiction, target):

![training visualization](./data/affordance/training_vis/000.png)

### 2e: Grasp prediction

In `affordance_model.py` we implement `AffordanceModel.predict_grasp` 
for both prediction and visualization.

### 2f: Evaluation on the training set

The model with the lowest loss should be saved as checkpoint at 
[data/affordance/best.ckpt](./data/affordance/best.ckpt)

Execute the following command to evaluate the model:

```shell
python3 eval.py --model affordance --task pick_training
```
Alternatively: 
```shell
sh scripts/run_2f_eval_train.sh
```

The prediction code has a success rate of 93.3%

Please click [here](https://drive.google.com/file/d/1dWeUKuE8m-GtMdkNEcTAUNQsTxbnP0WC/view)
for a video demo. Here is an example visualization from 
`data/affordance/eval_pick_training_vis/`:
![training visualization](./data/affordance/eval_pick_training_vis/YcbMustardBottle_1.png)





15/15 picked up; here is [a video demo]()


[2g](https://drive.google.com/file/d/1e8HI5OHMS88Ywdosk3it1a4uNcB6w38x/view)

[3a]()

[3b]()

[4c](https://drive.google.com/file/d/1Q9LcQxrtYUXTB1zSPpcEyC7mNL_jyKkc/view)


below is a visualization

![viz](./data/affordance/training_vis/000.png)