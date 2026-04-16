import numpy as np
import math

def orientation2euler(quaternion):

    w, x, y, z = quaternion
    
    # 检查四元数是否已归一化
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if abs(norm - 1.0) > 0.0001:
        # 归一化四元数
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    # 计算欧拉角（ZYX顺序，yaw-pitch-roll）
    # roll (x轴旋转)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # pitch (y轴旋转)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        # 处理万向节锁的情况
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)
    
    # yaw (z轴旋转)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return [roll, pitch, yaw]


def orientation2euler_deg(quaternion):
    euler_ori = orientation2euler(quaternion)
    return [euler_ori[0]*57.3, euler_ori[1]*57.3, euler_ori[2]*57.3]
