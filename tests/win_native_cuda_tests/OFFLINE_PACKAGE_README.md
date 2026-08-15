# LightGBM Windows 原生 CUDA 离线安装包

本目录是面向 **Windows x64 + CPython 3.11 + 单张 NVIDIA GPU** 的完整离线
wheelhouse，包含 LightGBM CUDA wheel 及其 Python 运行依赖。安装过程中不会访问
PyPI 或其他网络索引。

对应源码标签：[`v4.7.0.99-win-cuda132-sm89ptx-r1`](https://github.com/yuweiyang9611/LightGBM-WinNativeCUDA-SingleGPU/tree/v4.7.0.99-win-cuda132-sm89ptx-r1)。

## 兼容范围

内置 `lib_lightgbm.dll` 使用 CUDA Toolkit 13.2.51 构建，并同时包含：

* `sm_89` cubin：RTX 40 系列、RTX Ada、L4/L40/L40S 等 Compute Capability 8.9
  GPU 使用原生代码；RTX 4060 Laptop GPU + 驱动 610.88 已完成实机测试。
* `compute_89` PTX：可由足够新的驱动向 Compute Capability 8.9 或更高架构即时
  编译；更新架构属于前向兼容路径，除 `sm_89` 外尚未逐型号实测。

目标电脑必须具备：

* 对 RTX 40 / Ada 等 `sm_89` GPU，R580 是 CUDA 13.x cubin minor compatibility
  的理论下限；若使用 `compute_89` PTX fallback，需 R595 或更新驱动，并建议确认
  `nvidia-smi` 显示 CUDA 13.2 或更高。当前仅在驱动 610.88 上实测；
* 最新 Microsoft Visual C++ v14 x64 Redistributable（14.51 或更高），包括
  `VCOMP140.dll`；
* 64 位 CPython 3.11；
* 单卡运行，LightGBM 参数使用 `device_type="cuda"`、`num_gpu=1`。电脑可以安装
  多张显卡，但本构建一次只使用 `gpu_device_id` 选中的一张；`num_gpu>1` 会报错。

目标电脑不需要 CUDA Toolkit、CMake、Visual Studio、编译器或互联网连接。
`compute_89` PTX 不能向较旧架构反向兼容，因此 RTX 30 系列（`sm_86`）、A100
（`sm_80`）和 Jetson Orin（`sm_87`）不能使用本包。

## 安装

在当前解压目录打开 PowerShell，创建独立虚拟环境并运行：

```powershell
py -3.11 -m venv .venv
.\install_offline_wheel.ps1 -PythonExecutable ".\.venv\Scripts\python.exe"
```

安装脚本会先验证 `SHA256SUMS.txt` 中的每个文件，再使用 `pip --no-index` 安装。
等价的手动命令为：

```powershell
.\.venv\Scripts\python.exe -m pip install `
  --no-index `
  --find-links . `
  .\lightgbm-4.7.0.99-1cuda132sm89ptx-py3-none-win_amd64.whl
```

## 验证 CUDA 训练

```powershell
.\.venv\Scripts\python.exe .\verify_cuda_install.py
```

脚本必须以 `Windows native CUDA LightGBM verification passed` 结束。

---

## English quick start

This wheelhouse targets 64-bit Windows, CPython 3.11, and one NVIDIA GPU. It
contains an `sm_89` cubin plus `compute_89` PTX fallback. A driver version of
R580 is the theoretical minimum for the native `sm_89` cubin path. The PTX
fallback requires an R595-or-newer CUDA 13.2-capable driver; driver 610.88 was
tested. Microsoft Visual C++ v14 x64 Redistributable 14.51 or newer (including
`VCOMP140.dll`) is also required. The CUDA Toolkit and internet access are not.

From PowerShell in this extracted directory:

```powershell
py -3.11 -m venv .venv
.\install_offline_wheel.ps1 -PythonExecutable ".\.venv\Scripts\python.exe"
.\.venv\Scripts\python.exe .\verify_cuda_install.py
```
