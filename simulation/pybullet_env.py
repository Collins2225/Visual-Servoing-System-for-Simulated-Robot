import pybullet as p
import pybullet_data
import numpy as np
import cv2


class RobotEnv:
    def __init__(self, gui=True):
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        self.robot = p.loadURDF(
            "kuka_iiwa/model.urdf",
            basePosition=[0, 0, 0],
            useFixedBase=True
        )
        self.target = p.loadURDF(
            "sphere_small.urdf",
            basePosition=[0.5, 0.1, 0.5]
        )
        self.num_joints = p.getNumJoints(self.robot)
        self.ee_index = self.num_joints - 1
        home_angles = [0, -0.5, 0, -1.5, 0, 1.0, 0]
        for i, angle in enumerate(home_angles):
            p.resetJointState(self.robot, i, angle)
        print(f"Robot loaded with {self.num_joints} joints")
        print(f"End effector index: {self.ee_index}")

    def get_camera_image(self, width=640, height=480):
        ee_state = p.getLinkState(self.robot, self.ee_index)
        ee_pos = ee_state[4]
        ee_orn = ee_state[5]
        rot_flat = p.getMatrixFromQuaternion(ee_orn)
        rot_matrix = np.array(rot_flat).reshape(3, 3)
        camera_pos = np.array(ee_pos)
        camera_forward = rot_matrix[:, 2]
        camera_up = rot_matrix[:, 1]
        camera_target = camera_pos + camera_forward
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_pos,
            cameraTargetPosition=camera_target,
            cameraUpVector=camera_up
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=width / height,
            nearVal=0.1,
            farVal=10.0
        )
        _, _, rgb_pixels, _, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix
        )
        img_rgba = np.array(rgb_pixels, dtype=np.uint8).reshape(height, width, 4)
        img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
        return img_bgr

    def get_ee_pose(self):
        state = p.getLinkState(self.robot, self.ee_index)
        return state[4], state[5]

    def move_joints(self, joint_angles):
        for i, angle in enumerate(joint_angles):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=angle,
                force=500
            )
        p.stepSimulation()

    def compute_ik(self, target_pos, target_orn=None):
        if target_orn is None:
            target_orn = p.getQuaternionFromEuler([0, -np.pi, 0])
        joint_angles = p.calculateInverseKinematics(
            self.robot,
            self.ee_index,
            target_pos,
            target_orn
        )
        return joint_angles

    def reset_target(self, position):
        p.resetBasePositionAndOrientation(
            self.target,
            position,
            [0, 0, 0, 1]
        )

    def step(self):
        p.stepSimulation()

    def close(self):
        p.disconnect(self.client)
        print("Simulation closed.")
