# Codex 从零搭建 MuJoCo 机械仿真提示词

本文档面向其他 Codex 用户：他们通常只有 MuJoCo 环境、机械结构文件（STEP/STL/URDF/MJCF 草稿）和一些机械参数，不会拥有本仓库的 `README.md`、`Simulation.py`、`controller.py`、`kinematics.py`、`MJCF/scene.xml` 或 `tests/`。因此下面的提示词都按“从零生成项目骨架、再逐步部署和调试”的场景编写。

使用建议：先让 Codex 建项目骨架，再逐步加入机械模型、控制器、HUD、测试和调试工具。不要一次让 Codex 生成所有复杂控制逻辑。

## 1. 从零初始化项目

```text
我有 MuJoCo 环境和机械结构文件，但没有现成项目代码。请从零创建一个可运行的 MuJoCo 仿真项目。要求：
1. 生成 README.md，说明依赖、运行方式、模型来源、控制方式和测试方法；
2. 生成 requirements.txt，至少包含 mujoco、glfw、numpy、pytest；
3. 创建 MJCF/scene.xml，先用简化几何体搭出机体、轮子、腿部连杆、地面和测试地形；
4. 创建 Simulation.py，支持图形窗口、headless 验证、键盘控制、HUD 状态显示；
5. 创建 controller.py，先实现安全的基础力矩控制框架，不要一开始写过复杂控制；
6. 创建 kinematics.py，用于机械腿正逆运动学或占位接口；
7. 创建 tests/，至少包含模型加载、headless 步进、键盘命令、控制器输出范围测试；
8. 运行 python Simulation.py --headless --seconds 0.2 和 pytest，修到能通过。
```

## 2. 收集机械参数

```text
请先帮我列出部署 MuJoCo 机械仿真所需的参数清单，并根据我提供的信息建立参数表。需要包含：
1. 机器人总质量、质心估计、机体尺寸；
2. 轮半径、轮距、轮质量和轮轴方向；
3. 腿部机构类型，如串联腿、平行四边形、五连杆、四连杆；
4. 每个连杆长度、关节相对位置、关节旋转轴、关节限位；
5. 执行器类型：位置、速度、力矩、电机力矩上限、控制频率；
6. STEP/STL/URDF 文件路径和坐标系方向；
7. 地面、台阶、坡道、跳台等测试场景尺寸；
8. 目标功能：平衡、行走、转向、跳跃、爬台阶、抗扰动。
如果信息不足，请先生成可编辑的参数模板，不要瞎填不可验证的数值。
```

## 3. 生成 MJCF 场景

```text
请根据我的机械参数从零生成 MJCF/scene.xml。要求：
1. root body 至少释放 x/y/z/yaw/roll/pitch 自由度；
2. 轮子用 cylinder 或 capsule 碰撞体，不要直接用复杂 mesh 做主要接触；
3. 腿部连杆可先用 capsule/box 近似，之后再加 STL/STEP 显示网格；
4. 每个关节命名清晰，左右、前后、髋/膝/轮轴要一致；
5. actuator 先使用 motor 力矩执行器，并设置合理 gear/ctrlrange；
6. 地面和测试地形使用简单 box，包含平地、坡道、低台阶、高台阶；
7. 如果机械闭链暂时难建，可先用等效开链或 equality constraint，先保证模型能加载；
8. 写一个 tests/test_model.py，验证模型能加载、关键 body/joint/actuator 存在、仿真 100 步无 warning。
```

## 4. 导入 STEP/STL 作为显示网格

```text
我有 STEP/STL 机械外观文件，请帮我把它们作为 MuJoCo 显示网格接入，而不是直接作为主要碰撞。要求：
1. 创建 MJCF/meshes 或 MJCF/step_meshes 存放网格；
2. scene.xml 中 mesh geom 用于 visual group，碰撞仍保留简化几何；
3. 提供 detailed mesh 开关，例如 --detailed-meshes；
4. 记录网格来源、导出时间、文件 SHA-256 到 SOURCE.txt；
5. 如果 STEP 需要拆分，请生成 tools/export_step.py 或说明可用的 Gmsh/OpenCASCADE 流程；
6. 加测试确保快速几何和详细网格可切换，floor 和碰撞几何不被隐藏。
```

## 5. 建立运动学模块

```text
请为我的腿部机构生成 kinematics.py。要求：
1. 明确输入输出，例如关节角 -> 轮心/足端位置，目标腿长/摆角 -> 关节角；
2. 如果是五连杆或闭链机构，写出正解、逆解和雅可比；
3. 对不可达目标做 clip，并返回可解释的异常或状态；
4. 为几何参数建立 dataclass；
5. 增加 pytest：默认姿态、最小/最大腿长、雅可比数值差分一致性、左右腿镜像一致性；
6. 在 controller.py 中只调用 kinematics.py 的公共接口，不把几何公式散落在控制器里。
```

## 6. 搭建控制器骨架

```text
请从零生成 controller.py 的控制器骨架，先安全稳定，再逐步增加复杂控制。要求：
1. 定义 Command：speed、yaw、leg_length、leg_swing；
2. 定义 LegState：length、angle、length_rate、angle_rate；
3. 初始化姿态时设置关节到可达腿长，并清零速度和积分状态；
4. update(now) 中按固定控制周期执行：读取状态、接触检测、命令滤波、控制计算、力矩限幅、写入 data.ctrl；
5. 所有执行器都必须限幅和限速，避免第一帧爆力矩；
6. 先实现腿长 PD、轮速/速度阻尼、简单俯仰稳定；
7. 再逐步加入 VMC、LQR、偏航、跳跃、堵转恢复；
8. 每加一个功能都加测试，不要只靠目测图形仿真。
```

## 7. 添加 VMC/LQR 控制

```text
请为轮腿机器人添加 VMC + LQR 控制。要求：
1. 先说明状态量顺序，例如腿角、腿角速度、位置误差、速度误差、机体俯仰、俯仰角速度；
2. LQR_K 必须写清单位和输出含义：轮矩、虚拟腿摆矩 Tp；
3. VMC 将虚拟轴向力 F0 和虚拟摆矩 Tp 通过雅可比映射到关节力矩；
4. 轮矩、髋关节力矩、偏航辅助力矩都要有限幅和变化率限制；
5. 空中状态释放轮矩，保留必要的腿部姿态控制；
6. 停止、换向、反向恢复时重置位置目标，避免追旧目标疯车；
7. 提供测试：小俯仰恢复、前进速度跟踪、快速换向后停止、空中释放轮矩。
```

## 8. 图形仿真与 HUD

```text
请生成 Simulation.py 的图形仿真入口。要求：
1. 支持 --headless --seconds；
2. 图形模式使用 MuJoCo + GLFW；
3. 物理和渲染可以分离，渲染不能改变控制周期；
4. 键盘按住 W/S 或上下键控制前后，A/D 或左右键控制转向，Space 跳跃，X 停止；
5. HUD 实时显示：
   - keys held / last key；
   - cmd speed/yaw；
   - target speed/yaw；
   - leg target；
   - wheel/hip/yaw torque；
   - xyz、roll/pitch/yaw；
   - contact、airborne、stall、tip recovery、jump；
6. 图形窗口关闭后正确释放 glfw/context；
7. tests/test_simulation.py 验证键盘命令、headless 步进、HUD 文本构建。
```

## 9. 跳跃、台阶和恢复策略

```text
请为 MuJoCo 轮腿机器人添加台阶、跳跃和异常恢复策略。要求：
1. 先建立测试地形：低台阶、高台阶、坡道、无导入坡跳台；
2. 跳跃曲线分为收腿、爆发伸腿、空中收腿、落地恢复；
3. 跳台附近可使用专用起跳曲线，但普通地形不要过强弹跳；
4. 堵转检测使用：有速度命令、实际速度很低、俯仰变大、接触在台阶/跳台附近；
5. recovery 不要只清零轮矩，要区分：
   - 空中释放；
   - 零指令扶正；
   - 卡边收腿脱困；
   - 用户明确后退时的反向逃逸；
   - 原地转向大俯仰时限制轮矩；
6. 所有恢复状态必须能在 HUD 中显示 active/ready；
7. 加回归测试覆盖：不能无跳跃硬上高台阶、可以通过跳跃上台阶、卡边不会疯车、后退能退出。
```

## 10. 疯车/不可控问题诊断

```text
仿真出现疯车、原地乱跑、满轮矩不可控。请按以下流程诊断：
1. 先读取 HUD，而不是先改参数；
2. 对比 cmd speed/yaw 和 target speed/yaw 是否矛盾；
3. 查看 keys held / last key，确认用户到底按了什么；
4. 查看 pitch/roll/yaw 是否大角度；
5. 查看 wheel torque L/R、yaw assist 是否饱和；
6. 查看 contact、airborne、stall pose、tip recovery、jump 状态；
7. 判断根因：
   - 目标速度为 0 但轮矩满：位置误差或俯仰 LQR 没释放；
   - 用户后退但轮矩向前：恢复方向或 LQR 方向没有尊重用户；
   - 原地转向大俯仰仍满轮矩：需要 stall pose recovery；
   - 空中还给轮矩：接触/airborne 判断错误；
   - 长时间追旧目标：停止/换向时未重置 target_distance。
8. 用最小测试复现该状态，先写测试，再修控制器。
```

## 11. 用视频辅助调试

```text
我有录屏视频，请用它辅助调试 MuJoCo 仿真。请执行：
1. 找到视频文件；
2. 用 ffprobe 读取时长；
3. 用 ffmpeg 每 5 秒抽整段视频帧；
4. 对异常段每 1-2 秒加密抽帧；
5. 对比正常段和异常段 HUD；
6. 提取关键差异：按键、cmd/target、姿态、轮矩、接触、恢复状态；
7. 根据视频状态写回归测试；
8. 修复后运行 pytest 和 headless。
注意：如果用户环境没有本项目文件，请先让用户提供或生成 HUD 字段，否则视频只能判断姿态，无法定位控制链路。
```

## 12. 测试体系

```text
请为从零搭建的 MuJoCo 项目生成 pytest 测试体系。至少包含：
1. test_model_loads：MJCF 能加载；
2. test_required_names_exist：关键 body/joint/actuator/geom 存在；
3. test_headless_steps_without_warning：headless 步进无 warning；
4. test_keyboard_commands：键盘命令映射正确；
5. test_controller_output_limits：轮矩/关节力矩不超过限幅；
6. test_pitch_recovery：小俯仰扰动能恢复；
7. test_airborne_releases_wheels：空中释放轮矩；
8. test_stop_resets_target：停止/换向重置目标；
9. test_stall_recovery：卡边不会疯车；
10. test_jump_or_step：按项目目标测试跳跃/台阶。
```

## 13. 最终验收提示词

```text
请对这个从零搭建的 MuJoCo 机械仿真项目做最终验收：
1. 运行 python Simulation.py --headless --seconds 0.2；
2. 运行 pytest -q；
3. 如果有图形环境，启动 python Simulation.py；
4. 检查窗口、相机、HUD、键盘、接触、跳跃/台阶；
5. 报告：
   - 已生成哪些文件；
   - 如何运行；
   - 哪些测试通过；
   - 机械参数中哪些是估计值；
   - 哪些控制参数仍需实机或更精确模型校准；
   - 用户下一步应提供哪些机械/控制数据。
```

## 14. 给 Codex 的通用约束

```text
请遵守：
1. 不假设用户已有 README.md、Simulation.py、controller.py、kinematics.py、MJCF/scene.xml 或 tests/，需要从零创建；
2. 不把复杂 STEP/STL 网格直接作为主要碰撞；
3. 先让模型能加载和 headless 跑通，再加复杂控制；
4. 控制器每个保护状态都要可观测、可测试、可解释；
5. 修复疯车时不要只把轮矩置零，要区分扶正、后退逃逸、空中释放、卡边脱困；
6. 用户按键优先于过期自动恢复目标；
7. 每个功能都要有最小回归测试；
8. 修改后必须报告测试结果和残余风险。
```
