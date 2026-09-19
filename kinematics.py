"""平面五杆机构正逆运动学（单侧轮腿使用）。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class FiveBarGeometry:
    """五杆机构参数：髋距与各杆长度（单位：米）。"""
    hip_spacing: float = 0.088
    front_crank: float = 0.0833
    front_rod: float = 0.160
    rear_rod: float = 0.160
    rear_crank: float = 0.0833


class FiveBarKinematics:
    def __init__(self, geometry: FiveBarGeometry | None = None):
        # 如果未传参，使用默认几何尺寸
        self.g = geometry or FiveBarGeometry()

    @staticmethod
    def _circle_intersection(
        p0: np.ndarray, r0: float, p1: np.ndarray, r1: float, branch: int
    ) -> np.ndarray:
        # 两圆交点法：由两侧连杆构造末端点，branch决定取上/下分支
        delta = p1 - p0
        distance = float(np.linalg.norm(delta))
        if distance > r0 + r1 or distance < abs(r0 - r1) or distance == 0.0:
            raise ValueError("The requested five-bar pose is outside the workspace")
        along = (r0 * r0 - r1 * r1 + distance * distance) / (2.0 * distance)
        height = math.sqrt(max(r0 * r0 - along * along, 0.0))
        midpoint = p0 + along * delta / distance
        normal = np.array([-delta[1], delta[0]]) / distance
        return midpoint + math.copysign(height, branch) * normal

    def forward(self, front_angle: float, rear_angle: float, branch: int | None = None) -> np.ndarray:
        """根据前后曲柄角，求末端轮轴点C在髋中心平面内坐标。"""
        g = self.g
        a = np.array([-g.hip_spacing / 2.0, 0.0])
        e = np.array([g.hip_spacing / 2.0, 0.0])
        b = a + g.front_crank * np.array([math.cos(front_angle), math.sin(front_angle)])
        d = e + g.rear_crank * np.array([math.cos(rear_angle), math.sin(rear_angle)])
        if branch is not None:
            return self._circle_intersection(b, g.front_rod, d, g.rear_rod, branch)
        first = self._circle_intersection(b, g.front_rod, d, g.rear_rod, -1)
        second = self._circle_intersection(b, g.front_rod, d, g.rear_rod, +1)
        return first if first[1] < second[1] else second

    @staticmethod
    def _two_link_ik(
        root: np.ndarray, target: np.ndarray, first: float, second: float, elbow: int
    ) -> tuple[float, float]:
        # 两连杆逆解（夹角分支为+/-elbow），返回肩部与肘部角度
        vector = target - root
        distance = float(np.linalg.norm(vector))
        cosine = (distance * distance + first * first - second * second) / (2.0 * distance * first)
        if distance > first + second or distance < abs(first - second):
            raise ValueError("The requested target is outside the two-link workspace")
        offset = math.acos(float(np.clip(cosine, -1.0, 1.0)))
        shoulder = math.atan2(vector[1], vector[0]) + elbow * offset
        elbow_point = root + first * np.array([math.cos(shoulder), math.sin(shoulder)])
        second_angle = math.atan2(target[1] - elbow_point[1], target[0] - elbow_point[0])
        return shoulder, second_angle

    def inverse(
        self, length: float, virtual_angle: float = -math.pi / 2.0
    ) -> tuple[float, float, float, float]:
        """给定目标长度和虚拟角，返回前/后杆组的角度解。"""
        g = self.g
        target = length * np.array([math.cos(virtual_angle), math.sin(virtual_angle)])
        a = np.array([-g.hip_spacing / 2.0, 0.0])
        e = np.array([g.hip_spacing / 2.0, 0.0])
        # Keep l1 and l4 on the outside of their hip pivots, matching the
        # physical five-bar assembly instead of the crossed inner branch.
        front, front_rod = self._two_link_ik(a, target, g.front_crank, g.front_rod, -1)
        rear, rear_rod = self._two_link_ik(e, target, g.rear_crank, g.rear_rod, +1)
        return front, front_rod, rear, rear_rod

    def target_point(self, length: float, virtual_angle: float) -> np.ndarray:
        # 由虚拟腿长和虚拟角直接构造目标足点（仅用于参考）
        return length * np.array([math.cos(virtual_angle), math.sin(virtual_angle)])

    def jacobian(self, front_angle: float, rear_angle: float) -> np.ndarray:
        """根据回路几何，返回 dC/d(前曲柄角, 后曲柄角)。"""
        g = self.g
        a = np.array([-g.hip_spacing / 2.0, 0.0])
        e = np.array([g.hip_spacing / 2.0, 0.0])
        b = a + g.front_crank * np.array([math.cos(front_angle), math.sin(front_angle)])
        d = e + g.rear_crank * np.array([math.cos(rear_angle), math.sin(rear_angle)])
        c = self.forward(front_angle, rear_angle)
        db = g.front_crank * np.array([-math.sin(front_angle), math.cos(front_angle)])
        dd = g.rear_crank * np.array([-math.sin(rear_angle), math.cos(rear_angle)])
        constraints = np.vstack((c - b, c - d))
        inputs = np.array([(c - b) @ db, (c - d) @ dd])
        return np.linalg.solve(constraints, np.diag(inputs))
