# Habitat-Sim 0.2.4 RGB 与 PointNav RGB/depth 离屏验收

脚本：`habitat_rgb_offscreen_acceptance.py`。它使用官方 `habitat-test-scenes/skokloster-castle.glb` 与 PointNav 测试数据 `val` 第 0 个 episode 的固定起始位姿，创建一台 256×256、90° 水平视角、高 1.5 m 的针孔 RGB 相机。脚本只获取一帧，不执行导航动作。

## 当前 A800：配置检查与已通过的软件渲染

在现有 A800 容器中使用已有 Habitat-Sim 0.2.4 环境：

```bash
/opt/conda/envs/streamvln/bin/python -B \
  /srv/navigation/repos/rlinf-nav/scripts/habitat_rgb_offscreen_acceptance.py \
  --asset-root /srv/navigation/data \
  --check-config
```

预期退出码为 0，JSON 中 `mode="config_only"`、`camera_spec_count=1`、`renderer_created=false`、`visual_render_validated=false`。这验证 Python API、场景文件和配置构造；**不验证 EGL 或 RGB 渲染**。现有共享 A800 容器仍无 NVIDIA EGL，但项目私有源码构建已通过 Mesa llvmpipe 视觉验收，见下节。

### A800 私有 Mesa llvmpipe 实测

预编译 Habitat-Sim 0.2.4 在 Mesa EGL 下尝试选择 CUDA 设备而失败。项目私有源码 `repos/habitat-sim-cpu-egl` 加入仅由 `HABITAT_SIM_SOFTWARE_EGL=1` 启用的软件设备路径；私有 `envs/habitat-cpu-egl` 不修改共享 Python 或系统库。该环境下固定官方场景 RGB 单帧成功，Habitat 内部日志的渲染器为 `llvmpipe (LLVM 15.0.7, 256 bits)`。真实 Habitat-Lab RGB+米制 depth PointNav 也通过：6 步固定动作失败诊断及 56 步 oracle 成功正例分别生成 7/57 帧视频，均已全片解码并拉回本地。oracle Success=1、SPL=1；它使用仿真内部目标与位姿，不是学习策略。

[CPU EGL 验收结果](../../artifacts/habitat_cpu_egl/RESULT.md)保存补丁、私有构建/运行脚本、日志、PNG、depth 数组、视频、逐步轨迹和 SHA256 清单。远端原件在 `/srv/navigation/runs/habitat-cpu-egl`。复跑前应查看 `build.sh` 与 `run_rgbd_*.sh`，将其中输出目录换成**新的**目录；脚本会拒绝覆盖现有验收产物。软件渲染只在这个小场景单环境录制段达到约 10 步/秒，不能外推到大场景或训练吞吐。

## 具备 NVIDIA EGL 的 A800 新容器或 4090 Linux 节点

先确保节点安装 Habitat-Sim **0.2.4**、可访问同一官方测试场景，并能使用 NVIDIA 图形/EGL 库。容器须向进程提供图形驱动能力；原生 Linux 节点则使用本机驱动。用该节点的实际路径运行：

```bash
python -B /path/to/rlinf-nav/scripts/habitat_rgb_offscreen_acceptance.py \
  --asset-root /path/to/data \
  --gpu-device-id 0 \
  --output-png /path/to/videos/habitat_rgb_offscreen.png
```

`--gpu-device-id` 是进程可见的逻辑编号；若先用 `CUDA_VISIBLE_DEVICES=4` 只暴露一张卡，通常仍传 `0`。脚本不要求 `DISPLAY`。`MUJOCO_GL` 是 MuJoCo 的设置，与此脚本无关。

预期退出码为 0，标准输出和同名 `.json` 清单中含 `mode="rgb_single_frame"`、`renderer_created=true`、`visual_render_validated=true`、`frame_shape_hwc=[256,256,3]` 或 `[256,256,4]`、`frame_dtype="uint8"`，以及原始帧、PNG、场景的 SHA-256。PNG 旁生成 `habitat_rgb_offscreen.json`。脚本会拒绝尺寸错误或完全均匀的 RGB 帧。不同显卡和驱动的像素哈希可能不同；哈希用于记录与传输完整性，**不作跨设备固定 golden 值**。

若 `habitat_sim.Simulator(config)` 因 EGL、OpenGL 或驱动不可用而失败，此节点的视觉验收失败且不会生成成功清单。成功也仅证明该节点可从固定测试场景取得 RGB 帧，不证明 Habitat-Lab PointNav、RLinf PPO、长时间稳定性或视频编码。

## RGB + depth 的真实 PointNav 步进验收

配套脚本 [`habitat_rgbd_pointnav_acceptance.py`](habitat_rgbd_pointnav_acceptance.py) 使用 Habitat-Lab 0.2.4 的官方 `pointnav_habitat_test.yaml` 与 `val` episode 0 原始起点，同时开启 256×256 RGB 和以米为单位、裁剪到 0–10 m 的 depth。默认脚本策略固定执行 `move_forward, turn_left, move_forward, turn_right, move_forward, stop`；这是传感器与任务链路诊断，**不是学得的导航策略或成功率实验**。每一步都由真实 `habitat.Env.step()` 推进，并保存 PointGoal、官方 distance/Success/SPL/collision、位姿和两种原始帧的哈希。

在任意本地 Python（无 Habitat、数据集和 GPU 也可）先检查参数及预期文件布局：

```powershell
python -B C:\path\to\rlinf-nav\scripts\habitat_rgbd_pointnav_acceptance.py --check-config
```

`mode="static_check"`、`habitat_imported=false`、`renderer_created=false` 是预期值。若加 `--asset-root`，脚本仅检查数据集与 episode 0 测试场景是否存在；`dataset_exists=false` 仍不会把静态检查当成失败或视觉验收。此模式不导入 Habitat/Numpy，不启动 EGL 或 CUDA。

在已安装 Habitat-Lab 0.2.4 的节点，还可以运行 `--check-api --asset-root /path/to/data`。它构造官方任务配置并验证 RGB/depth、碰撞测量、米制 depth 与逻辑 GPU ID 的配置字段，**不创建 Simulator 或 renderer**；此检查仍不能证明 EGL、逐步观察或视频可用。2026-09-24 在当前 A800 的隔离项目副本中以 `navcloud` 运行通过：`sensor_keys=["depth_sensor","rgb_sensor"]`、`depth_normalized=false`、`collisions_measurement_type="Collisions"`、`gpu_device_id=0`、`gpu_gpu=false`、`renderer_created=false`。这是纯配置结果，没有占用 GPU 或生成视觉帧。

在新 GPU 容器或已核对身份、图形能力的 4090 Linux 节点，使用**新的、尚不存在的**输出目录：

```bash
python -B /path/to/rlinf-nav/scripts/habitat_rgbd_pointnav_acceptance.py \
  --asset-root /path/to/data \
  --gpu-device-id 0 \
  --output-dir /path/to/runs/rgbd-pointnav-episode0
```

需要 Habitat-Lab/Habitat-Sim **0.2.4**、官方 test-scenes + PointNav `val`、`ffmpeg` 和 `ffprobe`。`--gpu-device-id` 是进程可见的逻辑 GPU 编号。默认仅一个 episode、一个环境，视频是左 RGB、右 depth 灰度的 H.264 512×256、6 FPS 文件；原始 depth 全帧另存 `.npy`，避免 8 位预览丢失距离精度。输出目录含 `rgb_reset.png`、`depth_reset.png`、`depth_frames_meters.npy`、`rgb_depth.mp4`、`trace.jsonl` 和 `manifest.json`。清单记录数据/场景/脚本及各产物 SHA-256、传感器配置、STOP/Success/SPL/碰撞、`env.step()` 与视频完成编码后的吞吐，并要求 FFmpeg 完整解码、FFprobe 解码帧数等于 reset 加各步帧数。极短脚本动作的吞吐仅作链路诊断，不能外推大批量 RL 速度。

`--policy oracle --max-steps 256` 可另跑官方起点的最短路 follower；它读取仿真内部目标与位姿，清单会明确标记特权信息。无论哪种策略，只有渲染、逐步观察、PointNav 指标和视频解码均通过才写成功清单。A800 私有 Mesa 构建已通过上述两个实际视觉序列；静态/API 配置检查仍不构成视觉验收，未来 GPU 容器和 4090 仍需分别实跑。
