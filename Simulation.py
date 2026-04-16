import mujoco
import mujoco.viewer
import numpy as np
import time
from environment import *
from VMC import *
from keyboard import *
import math
from Controller import *
def main():
    
    TORQUE = 1  #为1时给力矩，为0是无力矩
    GBC486 = LegWheelRobot('MJCF/env.xml')
    i = 0
    t1 = 1
    t2 = 4
    t3 = 20
    vmc_r = leg_VMC()
    vmc_l = leg_VMC()
    keyboard = KeyboardController()


    # ==========================================
    # 【修复：在循环开始前初始化 cmd 变量】
    # ==========================================
    cmd = [0.0, 0.0, 0.0]

    while True:
        i = i + 1
        
        # 1. 执行仿真步
        GBC486.step()  
        
        # 2. 传感器数据获取
        if i % t1 == 0: 
            GBC486.sensor_read_data()
            
        # 3. 键盘控制指令输入获取
        if i % t3 == 0:
            cmd = keyboard.get_command()
            
        # 4. VMC计算与核心控制 (每 4ms 执行一次)
        if i % t2 == 0:
            # 正向运动学计算
            vmc_r.vmc_calc_pos(phi1=GBC486.joint_pos[0]+math.pi, phi4=GBC486.joint_pos[1], pitch=GBC486.euler[1], gyro=GBC486.gyro[1])
            vmc_l.vmc_calc_pos(phi1=GBC486.joint_pos[3]+math.pi, phi4=GBC486.joint_pos[2], pitch=-GBC486.euler[1], gyro=-GBC486.gyro[1])
            
            # 腿部虚拟弹簧 PD 控制
            target_L0 = 0.20  # 目标腿长
            kp_leg = 800.0    # 弹簧刚度
            kd_leg = 20.0     # 弹簧阻尼
            
            vmc_r.F0 = kp_leg * (target_L0 - vmc_r.L0) - kd_leg * vmc_r.d_L0
            vmc_l.F0 = kp_leg * (target_L0 - vmc_l.L0) - kd_leg * vmc_l.d_L0
            
            vmc_r.Tp = 0
            vmc_l.Tp = 0
            
            vmc_l.vmc_calc_torque()
            vmc_r.vmc_calc_torque()
            
            # 连接键盘与轮毂电机
            forward_torque = cmd[0] * 3.0  # 前进后退力度
            turn_torque = cmd[1] * 1.5     # 转向力度
            
            w_r = forward_torque - turn_torque
            w_l = forward_torque + turn_torque
            
            # 下发所有电机指令
            GBC486.wheel_torque = [w_r, w_l]
            GBC486.joint_torque = [vmc_r.torque_set[1], vmc_r.torque_set[0], vmc_l.torque_set[0], vmc_l.torque_set[1]]
            GBC486.actuator_set_torque()
if __name__ == '__main__':
    main()