# Windows 原生 CUDA 版本兼容范围（必读）

> [!IMPORTANT]
> 本仓库当前构建的产物是 **Windows x64、单 NVIDIA GPU、CUDA Toolkit 13.2.51、`sm_89` cubin + `compute_89` PTX fallback** 版本的 `lib_lightgbm.dll`。
> `sm_89` 原生路径在 **GeForce RTX 4060 Laptop GPU、NVIDIA 驱动 610.88** 上完成真实 CUDA 训练测试；测试程序还会强制禁用 cubin，单独验证 PTX JIT 路径。
> “单 GPU”表示每次训练只使用 `gpu_device_id` 选择的一张卡；电脑可以安装多张显卡，但本构建为 `USE_NCCL=OFF`，因此 `num_gpu>1` 会主动报错。

## CUDA 与驱动版本

| 项目 | 当前支持范围 |
| --- | --- |
| 构建时 CUDA Toolkit | **CUDA Toolkit 13.2.51**，这是当前已经完整编译和测试过的基线版本 |
| 目标电脑 CUDA Toolkit | **不要求安装**；DLL 运行时不依赖目标电脑上的 Toolkit。即使安装了 13.2 或更高版本，也不能单独作为兼容性判断依据 |
| 目标电脑 NVIDIA 驱动 | `sm_89` cubin 路径的 CUDA 13.x minor compatibility 理论下限是 **R580**；使用 `compute_89` PTX fallback 时不能依赖旧 minor 驱动，要求驱动支持 CUDA 13.2 / PTX 9.2，建议使用 **R595 或更高的最新驱动**并确认 `nvidia-smi` 显示 CUDA 13.2 或更高。当前实测版本为 610.88 |
| 使用高于 13.2 的 Toolkit 重新编译 | 尚未逐版本验证；升级 Toolkit 后应重新执行本仓库的完整构建与测试 |

较新的 NVIDIA 驱动可以运行由较旧 CUDA Toolkit 构建的应用。因此，目标电脑即使显示 CUDA 13.3、13.4 或更高版本，只要同时满足 **驱动、GPU Compute Capability 和 Windows Runtime 依赖**，仍可运行这份由 CUDA 13.2 构建的 DLL。NVIDIA 同时明确说明：minor version compatibility 的旧驱动路径不保证 PTX JIT，因此更新架构依赖 PTX fallback 时必须使用支持 CUDA 13.2 PTX 的驱动。详见 [NVIDIA CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html) 和 [CUDA Toolkit Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)。

> [!NOTE]
> `nvidia-smi` 顶部显示的 `CUDA Version` 表示当前驱动能够支持的最高 CUDA 版本，并不表示目标电脑已经安装了对应版本的 CUDA Toolkit。

## 当前 `sm_89` + `compute_89` PTX DLL 支持的显卡

当前 DLL 使用：

```text
CMAKE_CUDA_ARCHITECTURES=89-real;89-virtual
```

产物同时包含：

* **`sm_89` cubin**：Compute Capability 8.9 显卡直接运行，不需要 JIT
* **`compute_89` relocatable PTX**：保存在各个 CUDA device object 中；当目标机没有匹配的 cubin 时，由 NVIDIA 驱动即时编译并完成设备链接

`89-real` 明确生成 `sm_89` cubin，`89-virtual` 明确生成 `compute_89` PTX。不要使用 `CMAKE_CUDA_ARCHITECTURES=native` 制作需要 PTX fallback 的分发产物，因为 `native` 只保证为构建机器的 GPU 生成原生代码。

### 原生 cubin 路径：已实测并推荐

| 类别 | 支持的 GPU 型号 |
| --- | --- |
| GeForce RTX 40 桌面版 | RTX 4090、RTX 4080 / 4080 SUPER、RTX 4070 Ti / 4070 Ti SUPER、RTX 4070 / 4070 SUPER、RTX 4060 Ti、RTX 4060 |
| GeForce RTX 40 笔记本版 | RTX 4090 Laptop GPU、RTX 4080 Laptop GPU、RTX 4070 Laptop GPU、RTX 4060 Laptop GPU、RTX 4050 Laptop GPU |
| NVIDIA RTX Ada 专业卡 | RTX 6000 Ada、RTX 5000 Ada、RTX 4500 Ada、RTX 4000 Ada、RTX 4000 SFF Ada、RTX 2000 Ada |
| NVIDIA 数据中心卡 | NVIDIA L4、NVIDIA L40、NVIDIA L40S |

型号与 Compute Capability 的官方对应关系见 [NVIDIA CUDA GPU Compute Capability](https://developer.nvidia.com/cuda-gpus)。如果目标 GPU 的 Compute Capability 查询结果是 `8.9`，即可直接使用当前 DLL 中的 `sm_89` cubin。

### PTX JIT 路径：向更新架构兼容

`compute_89` PTX 可以由足够新的 NVIDIA 驱动 JIT 到 Compute Capability **8.9 或更高**的 GPU，包括：

| JIT 目标架构 | 典型 GPU | 当前验证状态 |
| --- | --- | --- |
| `sm_89` | RTX 40 系列、RTX Ada、L4/L40/L40S | 已通过强制 PTX JIT 训练测试 |
| `sm_90` | H100、H200、GH200 | 理论兼容，尚未在对应硬件上实测 |
| `sm_100` / `sm_103` | B200、GB200、B300、GB300 | 理论兼容，尚未在对应硬件上实测 |
| `sm_120` / `sm_121` | RTX 50 系列、RTX PRO Blackwell、GB10 | 理论兼容，尚未在对应硬件上实测 |

上表的 Hopper / Blackwell 条目只表示 CUDA device code 的 PTX ISA 理论前向兼容，
不构成对应整机平台对本 `win_amd64` wheel 的支持声明。目标系统仍必须是 Windows x64，
并具有该 GPU 的 Windows 驱动；GH200、GB200、GB300、GB10 等常见 ARM64 或 Linux
平台不能直接安装这个 Windows wheel。Windows 用户优先把 RTX 50 / RTX PRO
Blackwell 视为待实机验证的 PTX 候选。

PTX JIT 第一次加载时会增加编译时间，生成结果通常会进入 NVIDIA 驱动缓存；它也不会自动利用专属于更新架构的编译期优化。NVIDIA 对 PTX 前向兼容和 JIT 的说明见 [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)。

使用下面的命令检查 DLL 中所有 relocatable fatbin，而不只是最终的 executable fatbin：

```powershell
cuobjdump --all-fatbin --list-ptx .\lib_lightgbm.dll
```

当前产物应列出 `cuda_link.sm_89.ptx` 以及 LightGBM 各 CUDA translation unit 的 `.sm_89.ptx`，共 16 项。不要只使用 `cuobjdump --list-ptx` 判断；在 separable compilation 构建中，它默认只查看最终 executable fatbin，可能错误报告 `No PTX file found`。

可以在目标电脑上检查：

```powershell
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
```

例如：

```text
NVIDIA GeForce RTX 4060 Laptop GPU, 8.9, 610.88
```

> [!WARNING]
> `compute_89` PTX 只能向前 JIT，不能向较低的 Compute Capability 反向兼容。因此当前 DLL **仍不兼容** RTX 30 系列（`sm_86`）、A100（`sm_80`）和 Jetson Orin（`sm_87`）。如需支持这些显卡，必须额外编译相应的 cubin 或更低版本的 PTX。

目标电脑还需要安装最新的 Microsoft Visual C++ v14 x64 Redistributable（本产物使用 MSVC 14.51 链接，运行库应为 14.51 或更高），并确保能够找到 `MSVCP140.dll`、`VCRUNTIME140.dll`、`VCRUNTIME140_1.dll` 和 `VCOMP140.dll`。

---

NCCL 是 **NVIDIA Collective Communications Library**，中文可以理解为：

> **NVIDIA 集合通信库**

它主要用于 **多张 NVIDIA GPU 之间高速交换数据**。

在你前面问的 LightGBM CUDA 场景里，可以把它理解成：

```text
单 GPU：
CPU ↔ GPU
不需要 NCCL

多 GPU：
GPU 0 ↔ GPU 1 ↔ GPU 2 ...
          ↑
         NCCL
```

NCCL 提供的核心操作包括：

* `AllReduce`：把多张 GPU 上的数据求和/求平均后，再把结果发回每张 GPU
* `Broadcast`：一张 GPU 的数据发给所有 GPU
* `Reduce`
* `AllGather`
* `ReduceScatter`

例如 LightGBM 用两张 GPU 并行计算直方图：

```text
GPU 0
计算 histogram A
       \
        → NCCL AllReduce → 全局 histogram
       /
GPU 1
计算 histogram B
```

本质上就是：

```text
GlobalHistogram = Histogram_GPU0 + Histogram_GPU1
```

LightGBM 源码中的：

```cpp
NCCLAllReduce(...)
```

就是在做这种事情。

### NCCL 和 CUDA不是一回事

容易混淆的是：

```text
CUDA
├─ GPU 计算
├─ cudaMalloc
├─ cudaMemcpy
├─ CUDA Kernel
└─ 单 GPU 完全可以使用

NCCL
├─ 建立在 CUDA 之上
├─ GPU ↔ GPU 通信
└─ 主要服务于多 GPU / 多机器
```

所以：

> **CUDA = 让 GPU 做计算**
>
> **NCCL = 让多张 GPU 相互通信**

这也是为什么刚才说 **Windows 单卡 CUDA LightGBM 理论上可以不需要 NCCL**：一张显卡根本没有其他 GPU 需要通信。

LightGBM 现在的问题更像是：

```text
想用 CUDA
   ↓
LightGBM 构建系统却规定
   ↓
必须安装 NCCL
   ↓
NCCL 官方主要支持 Linux
   ↓
Windows 原生 CUDA 构建被卡住
```

而实际上对于：

```python
{
    "device_type": "cuda",
    "num_gpu": 1
}
```

NCCL基本没有存在的必要。

因此，**把 NCCL 从 LightGBM 的“CUDA 必选依赖”改成“多 GPU 可选依赖”**，就是让 Windows 原生 CUDA 成为可能的核心思路。


你构建出来的 Windows CUDA 版 LightGBM，最终给 Python 使用的核心就是这个文件：

```text
lib_lightgbm.dll
```

LightGBM 的 Python 包本质上是 Python 包装层，底层通过 `ctypes` 加载 `lib_lightgbm.dll`。当前源码会搜索 `lightgbm/lib/lib_lightgbm.dll` 等位置。

所以最推荐的流程是：

```text
你修改后的 LightGBM C++ / CUDA 源码
            ↓
         CMake
            ↓
     lib_lightgbm.dll
            ↓
   LightGBM Python package
            ↓
        .whl 文件
            ↓
       pip install
            ↓
import lightgbm as lgb
            ↓
device_type="cuda"
```

## 1. 先编译你的 Windows CUDA DLL

假设目录：

```text
C:\Dev\LightGBM\
```

你的修改已经加入：

```text
USE_CUDA=ON
USE_NCCL=OFF
```

编译：

```bat
cd C:\Dev\LightGBM

cmake -S . -B build ^
    -A x64 ^
    -DUSE_CUDA=ON ^
    -DUSE_NCCL=OFF ^
    -DCMAKE_CUDA_ARCHITECTURES="89-real;89-virtual"

cmake --build build --config Release -j
```

最终目标是得到：

```text
lib_lightgbm.dll
```

比如可能位于：

```text
C:\Dev\LightGBM\build\Release\lib_lightgbm.dll
```

具体输出路径取决于你修改后的 CMake 配置。

---

# 2. 最简单的测试方法：直接让 Python 加载这个 DLL

LightGBM 当前的 `libpath.py` 会寻找：

```text
lightgbm\
    lib\
        lib_lightgbm.dll
```

所以可以先：

```bat
pip install lightgbm
```

然后找 Python 安装位置：

```bat
python -c "import lightgbm; print(lightgbm.__file__)"
```

例如输出：

```text
C:\Users\xxx\miniconda3\envs\quant\Lib\site-packages\lightgbm\__init__.py
```

那么 DLL 就在：

```text
C:\Users\xxx\miniconda3\envs\quant\
    Lib\
      site-packages\
        lightgbm\
          lib\
            lib_lightgbm.dll
```

你可以先把官方 DLL 备份：

```text
lib_lightgbm.dll
        ↓
lib_lightgbm_original.dll
```

然后把自己编译的：

```text
C:\Dev\LightGBM\build\Release\lib_lightgbm.dll
```

复制进去。

这样：

```python
import lightgbm as lgb
```

实际上加载的就是你的 CUDA 版 DLL。

### 这个方法适合测试

但不适合长期使用，因为：

```text
Python wrapper 版本
       ↕
C++ DLL 版本
```

最好来自**完全相同的 LightGBM commit**，否则 C API 有变化时可能产生 ABI/API 不兼容。

---

# 3. 更规范的方法：制作你自己的 Python wheel

这才是我推荐的最终方案。

LightGBM 官方其实已经专门提供了一个：

```text
--precompile
```

功能。

用途就是：

> 已经单独编译好了 `lib_lightgbm.dll`，然后把这个 DLL 包进 Python package。

当前官方 `build-python.sh` 明确支持这一流程。

脚本会自动寻找：

```text
repo root/lib_lightgbm.dll
```

或者：

```text
repo root/Release/lib_lightgbm.dll
```

以及官方 Windows DLL 输出目录，然后复制到：

```text
lightgbm/lib/lib_lightgbm.dll
```

再制作 wheel。

所以你编译完成后，可以简单地：

```bat
copy build\Release\lib_lightgbm.dll lib_lightgbm.dll
```

此时：

```text
C:\Dev\LightGBM
│
├─ CMakeLists.txt
├─ build-python.sh
├─ python-package
├─ src
├─ include
│
└─ lib_lightgbm.dll      ← 你的 Windows CUDA 版本
```

然后使用 Git Bash：

```bash
cd /c/Dev/LightGBM

sh ./build-python.sh install --precompile
```

官方脚本会：

```text
lib_lightgbm.dll
       ↓
复制进 Python 包
       ↓
构建 wheel
       ↓
pip 安装 wheel
```

当前脚本的实际逻辑就是先建立：

```text
lightgbm/lib/
```

然后：

```text
lib_lightgbm.dll
      ↓
lightgbm/lib/lib_lightgbm.dll
```

最后调用 `python -m build` 创建 wheel，再执行 `pip install`。

官方文档也专门说明：如果动态库已经自己编译完成，可以执行 `build-python.sh install --precompile` 来安装使用该动态库的 Python package。([GitHub][1])

---

# 4. 安装完成后的结构

最终你的 Python 环境大致是：

```text
Python
│
└─ site-packages
   │
   └─ lightgbm
      │
      ├─ __init__.py
      ├─ basic.py
      ├─ sklearn.py
      ├─ engine.py
      │
      └─ lib
         │
         └─ lib_lightgbm.dll
                  ↑
                  │
          你编译的 CUDA 版本
```

Python 调用链：

```text
LGBMRegressor
      │
      ↓
lightgbm/sklearn.py
      │
      ↓
lightgbm/basic.py
      │
      ↓
ctypes
      │
      ↓
lib_lightgbm.dll
      │
      ↓
LightGBM C API
      │
      ↓
C++ GBDT
      │
      ↓
CUDA Tree Learner
      │
      ↓
CUDA Kernel
      │
      ↓
NVIDIA GPU
```

所以不需要自己写 Python CUDA binding。

**LightGBM 已经把 Python ↔ C++ 这一层做好了。**

---

# 5. Python 代码完全不需要特殊修改

例如：

```python
import lightgbm as lgb

model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=63,

    device_type="cuda",
    gpu_device_id=0,
)

model.fit(X_train, y_train)
```

或者原生 API：

```python
params = {
    "objective": "regression",
    "device_type": "cuda",
    "gpu_device_id": 0,
}

train_set = lgb.Dataset(
    X_train,
    label=y_train,
)

model = lgb.train(
    params,
    train_set,
    num_boost_round=1000,
)
```

官方当前 Python 文档也是通过参数：

```python
{"device": "cuda"}
```

来选择 CUDA 后端。([GitHub][1])

你的 Windows 移植成功之后，Python 层不需要知道“这是 Windows 上自己移植的 CUDA”。

---

# 6. 怎么确认确实用了你的 CUDA DLL

第一步：

```python
import lightgbm

print(lightgbm.__file__)
```

然后：

```python
from lightgbm.libpath import _find_lib_path

print(_find_lib_path())
```

应该看到类似：

```text
[
 'C:\\...\site-packages\\lightgbm\\lib\\lib_lightgbm.dll'
]
```

这与当前 LightGBM 的 DLL 搜索机制一致。

然后实际训练：

```python
import numpy as np
import lightgbm as lgb

X = np.random.random((100_000, 200))
y = np.random.random(100_000)

dataset = lgb.Dataset(X, label=y)

params = {
    "objective": "regression",
    "device_type": "cuda",
    "gpu_device_id": 0,
    "verbosity": 2,
}

model = lgb.train(
    params,
    dataset,
    num_boost_round=100,
)
```

如果是普通 Windows 官方版本，因为官方目前仍然不支持 Windows CUDA 构建，`device_type="cuda"` 不会得到你想要的 Windows CUDA 后端；官方 Windows wheel 默认支持的是 OpenCL `gpu`。([GitHub][1])

而你的版本如果移植成功，就应该真正进入 CUDA tree learner。

---

# 7. 还可以监控显卡

训练时另开 CMD：

```bat
nvidia-smi -l 1
```

如果 CUDA 工作正常，可以看到：

```text
python.exe
```

出现在 GPU process 列表中，同时显存使用增加。

不过这只能证明 Python 进程使用了 NVIDIA GPU；真正验证 LightGBM CUDA 后端，还是应该结合 LightGBM 日志和 `device_type="cuda"` 的成功训练结果。

---

# 8. 最终建议你制作一个自己的 wheel

例如：

```text
lightgbm-4.7.0.99-1cuda132sm89ptx-py3-none-win_amd64.whl
```

里面直接包含：

```text
lightgbm/
├─ basic.py
├─ sklearn.py
├─ ...
└─ lib/
   └─ lib_lightgbm.dll
       └─ Windows Native CUDA
```

之后你的任意 Python 虚拟环境只需要：

```bat
pip uninstall lightgbm -y

pip install lightgbm-4.7.0.99-1cuda132sm89ptx-py3-none-win_amd64.whl
```

然后：

```python
from lightgbm import LGBMRegressor

model = LGBMRegressor(
    device_type="cuda"
)
```

就能用。

**因此你真正要交付的不是单独一个 DLL，而是一个包含这个 DLL 的 Windows wheel。** 对你前面准备做的大量 LightGBM + Optuna + walk-forward 实验，这种方式也最干净：环境可以固定，卸载和重新安装都简单。

---

# 9. 使用已经生成的离线 pip 包

当前离线包面向 **Windows x64 + CPython 3.11**，其中包含：

* `lightgbm-4.7.0.99-1cuda132sm89ptx-py3-none-win_amd64.whl`
* `narwhals==2.24.0`
* `numpy==2.4.6` 的 CPython 3.11 Windows x64 wheel
* `scipy==1.17.1` 的 CPython 3.11 Windows x64 wheel
* `install_offline_wheel.ps1`、SHA-256 校验清单和 CUDA 验证脚本

将 ZIP 复制到目标电脑并解压。在解压目录打开 PowerShell：

```powershell
py -3.11 -m venv .venv
.\install_offline_wheel.ps1 -PythonExecutable ".\.venv\Scripts\python.exe"
```

安装脚本会先核对 `SHA256SUMS.txt`，然后强制使用：

```text
pip --no-index --find-links <当前离线包目录>
```

因此安装过程不会访问 PyPI 或其他网络索引。安装后执行：

```powershell
.\.venv\Scripts\python.exe .\verify_cuda_install.py
```

成功时应输出：

```text
Windows native CUDA LightGBM verification passed
```

这条 wheelhouse 安装路径的目标电脑不需要安装 CUDA Toolkit、CMake 或
Visual Studio，但仍需要兼容的 NVIDIA 驱动、CPython 3.11 和 Microsoft
Visual C++ x64 Runtime（包括 `VCOMP140.dll`）。

---

# 10. 使用零安装便携测试包

对于只有 Windows x64、兼容 NVIDIA GPU 和驱动的离线测试机，使用
Release 中的：

```text
lightgbm-<version>-<build>-win_amd64-cp311-portable.zip
```

它在离线 wheelhouse 之上额外包含：

* 官方 CPython 3.11 embeddable runtime；
* 直接放在 `python.exe` 同目录的 MSVC 14.51+ CRT 和 OpenMP DLL；
* 包内 SHA-256 完整性清单；
* GPU / Compute Capability / 驱动预检查与 CUDA 训练冒烟测试；
* JSON 结果和控制台日志。

解压到可写目录或 U 盘后，双击：

```text
START_TEST.cmd
```

结果位于：

```text
results\<machine-time>\result.json
results\<machine-time>\console.log
```

启动器不安装 Python、VC++ Runtime 或其他系统组件，不修改注册表和系统
`PATH`。Python 临时文件与 CUDA JIT 缓存限制在当次结果目录中，结束后自动
删除。保存结果后，删除整个解压目录即可清理应用自身产生的文件。

使用其他 GPU 编号：

```powershell
.\run_portable_test.ps1 -GpuDeviceId 1
```

强制验证 PTX JIT 路径：

```powershell
.\run_portable_test.ps1 -GpuDeviceId 0 -ForcePtxJit
```

便携包的可复现构建、第三方运行库来源和“不污染”边界说明见
[`packaging/windows_cuda_portable/README.md`](../packaging/windows_cuda_portable/README.md)。

[1]: https://github.com/lightgbm-org/LightGBM/blob/main/python-package/README.rst "LightGBM/python-package/README.rst at main · lightgbm-org/LightGBM · GitHub"
