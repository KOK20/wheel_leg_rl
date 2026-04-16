import math
from math import sin,cos

class leg_VMC:
    """
    虚拟机械腿参数类
    """
    def __init__(self):
        # 长度参数，单位为m
        self.l5 = 0.0  # AE距离
        self.l1 = 0.215
        self.l4 = 0.215
        self.l2 = 0.258
        self.l3 = 0.258
        
        # 坐标点
        self.XB = 0.0
        self.YB = 0.0
        self.XD = 0.0
        self.YD = 0.0
        
        # C点坐标
        self.XC = 0.0
        self.YC = 0.0
        self.L0 = 0.0      # C点的极径
        self.phi0 = 0.0    # C点的极角
        self.alpha = 0.0
        self.d_alpha = 0.0
        
        # 距离参数
        self.lBD = 0.0     # BD之间的距离
        
        # 角度变化率
        self.d_phi0 = 0.0
        self.last_phi0 = 0.0
        
        # 中间计算参数
        self.A0 = 0.0
        self.B0 = 0.0
        self.C0 = 0.0
        self.phi2 = 0.0
        self.phi3 = 0.0
        self.phi1 = 0.0
        self.phi4 = 0.0
        
        # 雅可比矩阵系数
        self.j11 = 0.0
        self.j12 = 0.0
        self.j21 = 0.0
        self.j22 = 0.0
        
        # 扭矩设置值（2个元素的列表）
        self.torque_set = [0.0, 0.0]
        
        # 力和扭矩参数
        self.F0 = 0.0
        self.Tp = 0.0
        self.F02 = 0.0
        
        # theta相关参数
        self.theta = 0.0
        self.d_theta = 0.0
        self.last_d_theta = 0.0
        self.dd_theta = 0.0
        
        # L0相关参数
        self.d_L0 = 0.0
        self.dd_L0 = 0.0
        self.last_L0 = 0.0
        self.last_d_L0 = 0.0

        # imu相关参数
        self.pitch = 0.0
        self.gyro = 0.0
        
        # 支撑力
        self.FN = 0.0
        
        # 标志位
        self.first_flag = 0
    
    def vmc_calc_pos(self, dt=0.004, phi1=None, phi4=None, pitch=None, gyro=None):

        # 使用参数或类属性
        if pitch is not None:
            self.pitch = pitch
        if gyro is not None:
            self.gyro = gyro
        if phi1 is not None:
            self.phi1 = phi1
        if phi4 is not None:
            self.phi4 = phi4
        
        # 从C代码中看，对于右腿，pitch和gyro需要取负号
        # 但这里保持通用性，根据实际需要调整
        PitchR = -self.pitch
        GyroR = -self.gyro
        
        # 计算B点和D点的坐标
        # D点坐标计算
        self.YD = self.l4 * math.sin(self.phi4)  # D点y坐标
        self.YB = self.l1 * math.sin(self.phi1)  # B点y坐标
        self.XD = self.l5 + self.l4 * math.cos(self.phi4)  # D点x坐标
        self.XB = self.l1 * math.cos(self.phi1)  # B点x坐标
        
        # 计算BD距离
        self.lBD = math.sqrt((self.XD - self.XB)**2 + (self.YD - self.YB)**2)
        
        # 计算phi2和phi3（基于几何约束）
        self.A0 = 2 * self.l2 * (self.XD - self.XB)
        self.B0 = 2 * self.l2 * (self.YD - self.YB)
        self.C0 = self.l2**2 + self.lBD**2 - self.l3**2
        
        # 计算phi2（使用atan2，注意分母为A0+C0）
        # 这里对应C代码中的公式：phi2 = 2*atan2f((B0 + sqrt(A0^2 + B0^2 - C0^2)), A0 + C0)
        discriminant = self.A0**2 + self.B0**2 - self.C0**2
        if discriminant < 0:
            # 避免数值误差导致负数
            discriminant = max(discriminant, 0)
        
        numerator = self.B0 + math.sqrt(discriminant)
        denominator = self.A0 + self.C0
        
        # 避免除零错误
        if abs(denominator) < 1e-12:
            self.phi2 = 0.0
        else:
            self.phi2 = 2 * math.atan2(numerator, denominator)
        
        # 计算phi3
        dy = self.YB - self.YD + self.l2 * math.sin(self.phi2)
        dx = self.XB - self.XD + self.l2 * math.cos(self.phi2)
        self.phi3 = math.atan2(dy, dx)
        
        # 计算C点直角坐标
        self.XC = self.XB + self.l2 * math.cos(self.phi2)
        self.YC = self.YB + self.l2 * math.sin(self.phi2)
        
        # 计算C点极坐标（以(l5/2, 0)为极点的极坐标）
        self.L0 = math.sqrt((self.XC - self.l5/2.0)**2 + self.YC**2)
        self.phi0 = math.atan2(self.YC, (self.XC - self.l5/2.0))

        # 计算alpha
        self.alpha = math.pi/2.0 - self.phi0
        
                
        """
        至此,轰轰烈烈的正向大运动学结束了
        """


        # 计算phi0的变化率（数值微分）
        if self.first_flag == 0:
            self.last_phi0 = self.phi0
            self.first_flag = 1

        self.d_phi0 = (self.phi0 - self.last_phi0) / dt
        self.d_alpha = -self.d_phi0
        
        # 计算theta和d_theta（状态变量，用于LQR控制）
        self.theta = math.pi/2.0 - PitchR - self.phi0
        self.d_theta = -GyroR - self.d_phi0
        
        # 更新last_phi0
        self.last_phi0 = self.phi0

        self.d_L0 = (self.L0-self.last_L0)/dt
        self.dd_L0 = (self.d_L0-self.last_d_L0)/dt

        self.last_d_L0 = self.d_L0
        self.last_L0 = self.L0

        self.dd_theta = (self.d_theta - self.last_d_theta)/dt
        self.last_d_theta = self.d_theta

    def vmc_calc_torque(self):
        sin_phi3_phi2 = math.sin(self.phi3 - self.phi2)


        # 计算j11
        self.j11 = (self.l1 * math.sin(self.phi0 - self.phi3) * 
                   math.sin(self.phi1 - self.phi2)) / sin_phi3_phi2
        
        
        self.j12 = (self.l1 * math.cos(self.phi0 - self.phi3) * 
                   math.sin(self.phi1 - self.phi2)) / (self.L0 * sin_phi3_phi2)
        
        self.j21 = (self.l4 * math.sin(self.phi0 - self.phi2) * 
                   math.sin(self.phi3 - self.phi4)) / sin_phi3_phi2
        
        self.j22 = (self.l4 * math.cos(self.phi0 - self.phi2) * 
                   math.sin(self.phi3 - self.phi4)) / (self.L0 * sin_phi3_phi2)
        
        self.torque_set[1] = self.j11 * self.F0 + self.j12 * self.Tp
        self.torque_set[0] = self.j21 * self.F0 + self.j22 * self.Tp

    #Tp：扭转力；F0：支持力


