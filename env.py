import pybullet as p
import pybullet_data
import numpy as np
import time

import camera
from assets.ycb_objects import getURDFPath
from control import get_movej_trajectory

class UR5PickEnviornment:
    def __init__(self, gui=True):
        # 0 load environment
        if gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._plane_id = p.loadURDF("plane.urdf")
        p.setGravity(0, 0, -9.8)

        # 1 load UR5 robot
        self.robot_body_id = p.loadURDF(
            "assets/ur5/ur5.urdf", [0, 0, 0.4], p.getQuaternionFromEuler([0, 0, 0]))
        self._mount_body_id = p.loadURDF(
            "assets/ur5/mount.urdf", [0, 0, 0.2], p.getQuaternionFromEuler([0, 0, 0]))

        # Get revolute joint indices of robot (skip fixed joints)
        robot_joint_info = [p.getJointInfo(self.robot_body_id, i) for i in range(
            p.getNumJoints(self.robot_body_id))]
        self._robot_joint_indices = [
            x[0] for x in robot_joint_info if x[2] == p.JOINT_REVOLUTE]

        # joint position threshold in radians (i.e. move until joint difference < epsilon)
        self._joint_epsilon = 1e-3

        # Robot home joint configuration (over tote 1)
        self.robot_home_joint_config = [
            -np.pi, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]
        # Robot goal joint configuration (over tote 1)
        self.robot_goal_joint_config = [
            0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]

        # 2 load tote
        # 3D workspace for tote 1
        self._workspace1_bounds = np.array([
            [0.38, 0.62],  # 3x2 rows: x,y,z cols: min,max
            [-0.22, 0.22],
            [0.00, 0.5]
        ])
        # 3D workspace for tote 2
        self._workspace2_bounds = np.copy(self._workspace1_bounds)
        self._workspace2_bounds[0, :] = - self._workspace2_bounds[0, ::-1]        # Load totes and fix them to their position
        # Load totes and fix them to their position
        self._tote1_position = (
            self._workspace1_bounds[:, 0] + self._workspace1_bounds[:, 1]) / 2
        self._tote1_position[2] = 0.01
        self._tote1_body_id = p.loadURDF(
            "assets/tote/toteA_large.urdf", self._tote1_position, p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)

        self._tote2_position = (
            self._workspace2_bounds[:, 0] + self._workspace2_bounds[:, 1]) / 2
        self._tote2_position[2] = 0.01
        self._tote2_body_id = p.loadURDF(
            "assets/tote/toteA_large.urdf", self._tote2_position, p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)

        # 3 load gripper
        self.robot_end_effector_link_index = 9
        self._robot_tool_offset = [0, 0, -0.05]
        # Distance between tool tip and end-effector joint
        self._tool_tip_to_ee_joint = np.array([0, 0, 0.15])

        # Attach robotiq gripper to UR5 robot
        # - We use createConstraint to add a fixed constraint between the ur5 robot and gripper.
        self._gripper_body_id = p.loadURDF("assets/gripper/robotiq_2f_85.urdf")
        p.resetBasePositionAndOrientation(self._gripper_body_id, [
                                          0.5, 0.1, 0.2], p.getQuaternionFromEuler([np.pi, 0, 0]))

        p.createConstraint(self.robot_body_id, self.robot_end_effector_link_index, self._gripper_body_id, 0, jointType=p.JOINT_FIXED, jointAxis=[
                           0, 0, 0], parentFramePosition=[0, 0, 0], childFramePosition=self._robot_tool_offset, childFrameOrientation=p.getQuaternionFromEuler([0, 0, np.pi/2]))

        # Set friction coefficients for gripper fingers
        for i in range(p.getNumJoints(self._gripper_body_id)):
            p.changeDynamics(self._gripper_body_id, i, lateralFriction=1.0, spinningFriction=1.0,
                             rollingFriction=0.0001, frictionAnchor=True)
        
        self.set_joints(self.robot_home_joint_config)

        # 4 load camera
        self.camera = camera.Camera(
            image_size=(128, 128),
            near=0.01,
            far=10.0,
            fov_w=80
        )
        camera_target_position = (self._workspace1_bounds[:, 0] + self._workspace1_bounds[:, 1]) / 2
        camera_target_position[2] = 0
        camera_distance = np.sqrt(((np.array([0.5, -0.5, 0.5]) - camera_target_position)**2).sum())
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=camera_target_position,
            distance=camera_distance,
            yaw=90,
            pitch=-90,
            roll=0,
            upAxisIndex=2,
        )

        # 5. prepare loading objects
        self.object_ids = list()
    
    def camera_to_world(self, cam_coords):
        pose = camera.cam_view2pose(self.view_matrix)
        world_coords = cam_coords @ pose[:3,:3].T + pose[:3,3]
        return world_coords

    def pixel_to_world(self, img_x, img_y, depth):
        """
        CV Coordinte Convension
        """
        intrinsics = self.camera.intrinsic_matrix
        fx = intrinsics[0,0]
        fy = intrinsics[1, 1]
        cx, cy = intrinsics[:2,2]
        x = (img_x - cx) * depth / fx
        y = (img_y - cy) * depth / fy
        cam_coords = np.array([x,y,depth], dtype=np.float32)
        world_coords = self.camera_to_world(cam_coords)
        return world_coords

    def image_pose_to_pick_pose(self, coord, angle, depth_obs, min_z=0.022):
        depth = depth_obs[coord[::-1]]
        world_coord = self.pixel_to_world(*coord, depth)
        world_coord[-1] = max(min_z, world_coord[-1]-0.05)
        world_angle = (-angle-90)/180*np.pi
        return world_coord, world_angle

    def load_ycb_objects(self, name_list, seed=None):
        rs = np.random.RandomState(seed=seed)
        for name in name_list:
            urdf_path = getURDFPath(name)
            position, orientation = self.get_random_pose(rs)
            obj_id = p.loadURDF(urdf_path, 
                position, p.getQuaternionFromEuler(orientation))
            self.object_ids.append(obj_id)
        self.step_simulation(1e3)

    def observe(self):
        rgb_obs, depth_obs, mask_obs = camera.make_obs(self.camera, self.view_matrix)
        return rgb_obs, depth_obs, mask_obs
    
    def get_random_pose(self, rs):
        low = self._workspace1_bounds[:,0].copy()
        low[-1] += 0.2
        high = self._workspace1_bounds[:,1].copy()
        high[-1] += 0.2
        position = rs.uniform(low, high, size=3)
        orientation = rs.uniform(-np.pi, np.pi,size=3)
        return position, orientation

    def reset_objects(self, seed=None):
        rs = np.random.RandomState(seed=seed)
        for obj_id in self.object_ids:
            position, orientation = self.get_random_pose(rs)
            p.resetBasePositionAndOrientation(
                obj_id, position, p.getQuaternionFromEuler(orientation))
        self.step_simulation(1e3)
    
    def remove_objects(self):
        for obj_id in self.object_ids:
            p.removeBody(obj_id)
        self.object_ids = list()
    
    def set_joints(self, target_joint_state, steps=1e2):
        assert len(self._robot_joint_indices) == len(target_joint_state)
        for joint, value in zip(self._robot_joint_indices, target_joint_state):
            p.resetJointState(self.robot_body_id, joint, value)
        if steps > 0:
            self.step_simulation(steps)

    def num_object_in_tote1(self):
        num_in = 0
        low = self._workspace1_bounds[:,0].copy()
        low -= 0.2
        high = self._workspace1_bounds[:,1].copy()
        high += 0.2

        for object_id in self.object_ids:
            pos, _ = p.getBasePositionAndOrientation(object_id)
            pos = np.array(pos)
            is_in = (low < pos).all()
            is_in &= (pos < high).all()
            if is_in:
                num_in += 1
        return num_in

    def move_joints(self, target_joint_state, acceleration=10, speed=3.0):
        """
            Move robot arm to specified joint configuration by appropriate motor control
        """
        assert len(self._robot_joint_indices) == len(target_joint_state)
        dt = 1./240
        q_current = np.array([x[0] for x in p.getJointStates(self.robot_body_id, self._robot_joint_indices)])
        q_target = np.array(target_joint_state)
        q_traj = get_movej_trajectory(q_current, q_target, 
            acceleration=acceleration, speed=speed)
        qdot_traj = np.gradient(q_traj, dt, axis=0)
        p_gain = 1 * np.ones(len(self._robot_joint_indices))
        d_gain = 1 * np.ones(len(self._robot_joint_indices))

        for i in range(len(q_traj)):
            p.setJointMotorControlArray(
                bodyUniqueId=self.robot_body_id, 
                jointIndices=self._robot_joint_indices,
                controlMode=p.POSITION_CONTROL, 
                targetPositions=q_traj[i],
                targetVelocities=qdot_traj[i],
                positionGains=p_gain,
                velocityGains=d_gain
            )
            self.step_simulation(1)

    def move_tool(self, position, orientation, acceleration=10, speed=3.0):
        """
            Move robot tool (end-effector) to a specified pose
            @param position: Target position of the end-effector link
            @param orientation: Target orientation of the end-effector link
        """
        target_joint_state = np.zeros((6,))  # this should contain appropriate joint angle values
        # ========= TODO: Part 1 ========
        # Using inverse kinematics (p.calculateInverseKinematics), find out the target joint configuration of the robot
        # in order to reach the desired end_effector position and orientation
        # HINT: p.calculateInverseKinematics takes in the end effector **link index** and not the **joint index**. You can use 
        #   self.robot_end_effector_link_index for this 
        # HINT: You might want to tune optional parameters of p.calculateInverseKinematics for better performance
        # ===============================
        target_joint_state[:] = p.calculateInverseKinematics(
            self.robot_body_id, self.robot_end_effector_link_index, position, orientation, maxNumIterations=100)
        self.move_joints(target_joint_state, acceleration=acceleration, speed=speed)

    def robot_go_home(self, speed=3.0):
        self.move_joints(self.robot_home_joint_config, speed=speed)

    def close_gripper(self):
        p.setJointMotorControl2(
            self._gripper_body_id, 1, p.VELOCITY_CONTROL, targetVelocity=5, force=10000)
        self.step_simulation(4e2)

    def open_gripper(self):
        p.setJointMotorControl2(
            self._gripper_body_id, 1, p.VELOCITY_CONTROL, targetVelocity=-5, force=10000)
        self.step_simulation(4e2)

    def check_grasp_success(self):
        return p.getJointState(self._gripper_body_id, 1)[0] < 0.834 - 0.001

    def execute_grasp(self, grasp_position, grasp_angle):
        """
            Execute grasp sequence
            @param: grasp_position: 3d position of place where the gripper jaws will be closed
            @param: grasp_angle: angle of gripper before executing grasp from positive x axis in radians 
        """
        # Adjust grasp_position to account for end-effector length
        grasp_position = grasp_position + self._tool_tip_to_ee_joint
        gripper_orientation = p.getQuaternionFromEuler(
            [np.pi, 0, grasp_angle])
        pre_grasp_position_over_bin = grasp_position+np.array([0, 0, 0.3])
        pre_grasp_position_over_object = grasp_position+np.array([0, 0, 0.1])
        post_grasp_position = grasp_position+np.array([0, 0, 0.3])
        grasp_success = False
        # ========= PART 2============
        # TODO: Implement the following grasp sequence:
        # 1. open gripper
        # 2. Move gripper to pre_grasp_position_over_bin
        # 3. Move gripper to pre_grasp_position_over_object
        # 4. Move gripper to grasp_position
        # 5. Close gripper
        # 6. Move gripper to post_grasp_position
        # 7. Move robot to robot_home_joint_config
        # 8. Detect whether or not the object was grasped and return grasp_success
        # ============================
        # 1. open gripper
        self.open_gripper()
        # 2. Move gripper to pre_grasp_position_over_bin
        self.move_tool(pre_grasp_position_over_bin, gripper_orientation)
        # 3. Move gripper to pre_grasp_position_over_object
        # self.move_tool(pre_grasp_position_over_object, gripper_orientation)
        # 4. Move gripper to grasp_position
        self.move_tool(grasp_position, gripper_orientation)
        # 5. Close gripper
        self.close_gripper()
        # 6. Move gripper to post_grasp_position
        self.move_tool(post_grasp_position, gripper_orientation)
        # 7. Move robot to robot_home_joint_config
        self.robot_go_home()
        # 8. Detect whether or not the object was grasped and return grasp_success
        grasp_success = self.check_grasp_success()

        return grasp_success

    def execute_place(self):
        self.move_joints([
            0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0], speed=6.0)
        self.open_gripper()
        self.robot_go_home(speed=6.0)
        
    def step_simulation(self, num_steps):
        for i in range(int(num_steps)):
            p.stepSimulation()
            if self._gripper_body_id is not None:
                # Constraints
                gripper_joint_positions = np.array([p.getJointState(self._gripper_body_id, i)[
                                                0] for i in range(p.getNumJoints(self._gripper_body_id))])
                p.setJointMotorControlArray(
                    self._gripper_body_id, [6, 3, 8, 5, 10], p.POSITION_CONTROL,
                    [
                        gripper_joint_positions[1], -gripper_joint_positions[1], 
                        -gripper_joint_positions[1], gripper_joint_positions[1],
                        gripper_joint_positions[1]
                    ],
                    positionGains=np.ones(5)
                )
