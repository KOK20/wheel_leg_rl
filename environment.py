import mujoco
import mujoco.viewer
import numpy as np
import time
import math
from caculation import *

class LegWheelRobot:
    """腿轮机器人仿真类"""
    
    def __init__(self, model_path: str = 'legwheel_robot1.xml'):
        # 加载模型
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.sensor_T = 0.001
        self.sensor_f = 1/self.sensor_T 
        self.wheel_r = 0.77

        self.gyro = []
        self.accel = []
        self.orien = []
        self.euler = []

        self.joint_pos = []
        self.wheel_vel = [0,0]

        self.x = 0 #整车位移
        self.d_x = 0 #整车速度

        self.sensor_data = []


        self.left_wheel_pos = 0
        self.right_wheel_pos = 0
        
        self.last_left_wheel_pos = 0
        self.last_right_wheel_pos = 0

        self.wheel_torque = [0,0]#顺序：右、左
        self.joint_torque = [0,0,0,0]#顺序：右前、右后、左前、左后



        # 启动可视化界面
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        print("MuJoCo界面已启动！按ESC退出")
    
    def sensor_read_data(self):
        """读取传感器数据"""
        # 更新传感器数据
        mujoco.mj_forward(self.model, self.data)
        
        # 四元数+欧拉角
        self.orien = self.data.sensor('orientation').data.copy()
        self.euler = orientation2euler(self.orien)
        # 陀螺仪（角速度）
        self.gyro = self.data.sensor('gyro').data.copy()
        # 轮速（官方给的轮速好像有问题，这边直接用当前位置与上一次位置做差，实车还是要用LK电机的轮速数据）
        self.right_wheel_pos = self.data.sensor('Right_Wheel_pos').data.copy()[0]
        self.left_wheel_pos =  self.data.sensor('Left_Wheel_pos').data.copy()[0]
        self.wheel_vel[0] = round((float)(self.right_wheel_pos - self.last_right_wheel_pos) * self.sensor_f,3)
        self.wheel_vel[1] = -round((float)(self.left_wheel_pos - self.last_left_wheel_pos  ) * self.sensor_f,3)
        self.last_right_wheel_pos = self.right_wheel_pos
        self.last_left_wheel_pos = self.left_wheel_pos
        
        self.d_x = (self.wheel_vel[0] + self.wheel_vel[1]) * 0.5 * 2*math.pi*self.wheel_r / 60
        self.x = self.x + self.d_x*self.sensor_T

        # 右前关节位置
        right_front_pos = self.data.sensor('Right_front_joint_pos').data.copy()[0]+0.027  #AB
        # 右后关节位置
        right_rear_pos = self.data.sensor('Right_rear_joint_pos').data.copy()[0]+1.3     #AG
        # 左前关节位置
        left_front_pos = self.data.sensor('Left_front_joint_pos').data.copy()[0]+0.003   #IJ
        # 左后关节位置
        left_rear_pos = self.data.sensor('Left_rear_joint_pos').data.copy()[0]-1.3       #IO
        self.joint_pos = np.array([right_front_pos, right_rear_pos, left_front_pos, left_rear_pos])
        

    def actuator_set_torque(self):
        """设置执行器力矩"""
        # 设置关节力矩
        self.data.ctrl[0] = self.joint_torque[0]  # 右前关节
        self.data.ctrl[1] = self.joint_torque[1]  # 右后关节
        self.data.ctrl[2] = self.joint_torque[2]  # 左前关节
        self.data.ctrl[3] = self.joint_torque[3]  # 左后关节
        
        # 设置轮子力矩
        self.data.ctrl[4] = self.wheel_torque[0]  # 右轮
        self.data.ctrl[5] = self.wheel_torque[1]  # 左轮（注意gainprm为-1）

    def set_joint_positions(self, joint_angles):

        # 获取关节索引
        joint_indices = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'jAG'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'jGH'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'jIO'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'jOP')
        ]
        
        # 设置关节位置和速度
        for i, idx in enumerate(joint_indices):
            if idx != -1 and i < len(joint_angles):
                self.data.qpos[idx] = joint_angles[i]
                self.data.qvel[idx] = 0.0  # 只重置关节速度
                # print(idx)
        
        # 更新模型状态
        mujoco.mj_forward(self.model, self.data)

    def step(self):
        """执行一步仿真"""
        mujoco.mj_step(self.model, self.data)
        self.viewer.sync()
    
    def reset(self):
        """重置机器人状态"""
        mujoco.mj_resetData(self.model, self.data)
        # self.motor_set_torque(0.0, 0.0)

