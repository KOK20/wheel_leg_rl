# WSL MuJoCo 仿真生成器 Demo

这个目录是独立功能，不依赖当前轮腿机器人的 `controller.py` 或 `Simulation.py`。

目标：

- Windows 浏览器作为前端
- WSL Python 作为后端
- 前端传入机械结构 JSON 与 Python 控制函数
- 后端生成 MJCF，运行 MuJoCo 头less 仿真
- 前端显示图形化动画回放、仿真摘要、曲线、生成的 XML 和下载链接
- 前端也可以请求 WSL 直接启动完整 MuJoCo 图形窗口，包括当前轮腿 Demo 和用户生成模型

## 启动

在 WSL 项目根目录运行：

```bash
.venv/bin/python wsl_sim_builder/server.py --host 0.0.0.0 --port 18765
```

然后在 Windows 浏览器打开：

```text
http://localhost:18765
```

如果 WSL 端口没有自动转发，可在 WSL 中查看 IP：

```bash
hostname -I
```

然后在 Windows 打开：

```text
http://WSL_IP:18765
```

## API

生成 MJCF：

```http
POST /api/generate
```

生成并运行：

```http
POST /api/run
```

启动完整 MuJoCo 图形 GUI：

```http
POST /api/launch/wheel-leg
POST /api/launch/generated
```

`/api/launch/wheel-leg` 会直接启动项目现有的 `Simulation.py`，用于测试 Windows 端部署到 WSL 后能否拉起真实图形仿真窗口。

`/api/launch/generated` 会先根据当前结构 JSON 和控制函数生成 MJCF，再启动 `wsl_sim_builder/gui_runner.py` 打开 MuJoCo viewer。

请求体格式：

```json
{
  "duration": 6.0,
  "sample_hz": 90,
  "structure": {
    "name": "motor_pendulum_demo",
    "timestep": 0.002,
    "gravity": [0, 0, -9.81],
    "bodies": [],
    "actuators": []
  },
  "control_code": "def control(t, qpos, qvel, ctrl, state):\n    ctrl['motor'] = 0.0\n"
}
```

控制函数签名：

```python
def control(t, qpos, qvel, ctrl, state):
    ...
```

可用变量：

- `t`：当前仿真时间
- `qpos`：按关节名索引的位置字典
- `qvel`：按关节名索引的速度字典
- `ctrl`：按执行器名索引的控制字典，可以原地修改
- `state["body_pos"]`：按 body 名索引的世界坐标
- `math`、`np`：控制函数可直接使用

## 注意

这个 demo 对用户传入的 Python 控制函数只做了轻量限制，适合本机可信代码调试。
如果要开放给不可信用户，需要把控制函数放到独立容器或更严格的沙箱中运行。
