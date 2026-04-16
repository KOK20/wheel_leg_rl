from pynput import keyboard
import threading
import mujoco
import mujoco.viewer
import numpy as np
import time
class KeyboardController:
    def __init__(self):
        self.command = np.zeros(3)  # 示例：3维控制指令
        self.listener = None
        self._init_keyboard()
    
    def _on_press(self, key):
        # 抛弃 WASD，全面换用 Arrow Keys (方向键)
        if key == keyboard.Key.up:
            self.command[0] = 1.0  
        elif key == keyboard.Key.down:
            self.command[0] = -1.0  
        elif key == keyboard.Key.left:
            self.command[1] = -1.0  
        elif key == keyboard.Key.right:
            self.command[1] = 1.0  
        elif key == keyboard.Key.space:
            self.command[2] = 1.0  
    
    def _on_release(self, key):
        # 释放按键时重置指令
        if key in [keyboard.Key.up, keyboard.Key.down]:
            self.command[0] = 0.0
        elif key in [keyboard.Key.left, keyboard.Key.right]:
            self.command[1] = 0.0
        elif key == keyboard.Key.space:
            self.command[2] = 0.0
            
    def _init_keyboard(self):
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release)
        self.listener.start()
    
    def get_command(self):
        return self.command.copy()

if __name__ == '__main__':
    # 在仿真循环中使用
    controller = KeyboardController()
    model = mujoco.MjModel.from_xml_path("./MJCF/env.xml")
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running:
            # 获取键盘指令
            cmd = controller.get_command()
            # print(cmd)
            # 将指令应用到控制器
            data.ctrl[0] = cmd[0] * 10.0  # 前向力
            data.ctrl[1] = cmd[1] * 5.0   # 转向
            
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)