选择mujoco，给你舒适的仿真调车环境
本项目为串联腿空白仿真环境，预装VMC解算，可以轻松部署控制算法

环境：python3.10，ubuntu22.04

前置：
    conda create -n py310 python=3.10
    conda activate py310
    pip3 install mujoco==3.4.0
    pip3 install pynput

运行：
    python3 Simulation.py