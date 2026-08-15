# LightGBM Windows native CUDA portable test

这是 Windows x64 原生单 GPU CUDA 自包含测试包。测试机不需要安装
Python、pip、Microsoft Visual C++ Redistributable、CUDA Toolkit、CMake 或
Visual Studio，也不需要网络和管理员权限。

## 使用

1. 在解压目录双击 `START_TEST.cmd`。
2. 等待窗口显示 `PASS - CUDA training completed.`。
3. 查看 `results\<machine-time>\result.json` 和 `console.log`。
4. 复制出需要保存的结果，然后删除整个解压目录。

使用其他 GPU 编号：

```powershell
.\run_portable_test.ps1 -GpuDeviceId 1
```

强制测试 PTX JIT：

```powershell
.\run_portable_test.ps1 -GpuDeviceId 0 -ForcePtxJit
```

## 兼容范围

* Windows x64，每次训练一张 NVIDIA GPU。
* Compute Capability 8.9 使用 `sm_89` cubin，需要 R580 或更新驱动。
* CC > 8.9 使用 `compute_89` PTX JIT，需要 R595 或更新驱动。
* CC < 8.9（例如 RTX 30 / `sm_86` 和 A100 / `sm_80`）不兼容。

脚本不修改注册表、系统 `PATH` 或用户 Python 环境。可控的临时文件和
CUDA JIT 缓存都在包目录中，测试结束后自动清理。Windows 事件日志、
Defender/EDR 遥测等系统级记录不在应用的控制范围内。

---

This is a self-contained Windows x64, single-GPU CUDA test bundle. Extract it
to a writable directory, double-click `START_TEST.cmd`, and inspect the JSON
and console log under `results`. No Python, VC++ Redistributable, CUDA Toolkit,
network access, installation, or administrator rights are required.
