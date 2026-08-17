# LightGBM Windows CUDA 便携测试包

这套脚本把当前 Release 的 CPython 3.11 离线 wheelhouse 重新打包成真正的
Windows x64 自包含测试目录。目标机只需要兼容的 NVIDIA GPU 和驱动，
不需要安装 Python、pip、VC++ Redistributable、Visual Studio、CMake 或 CUDA
Toolkit，也不需要管理员权限。

> 注意：当前 GitHub 上的 `cp311-offline.zip` 是离线 wheelhouse，而不是自包含
> Python 运行时。它仍要求目标机预装 64 位 Python 3.11 和 VC++ 14.51。

## 产物结构

```text
LightGBM-CUDA-Portable/
├─ START_TEST.cmd                 # 双击入口
├─ run_portable_test.ps1         # 校验、隔离、清理和日志
├─ README.md                     # 目标机中英文快速说明
├─ THIRD_PARTY_NOTICES.md         # 第三方运行库来源
├─ SHA256SUMS.txt                # 包内不可变文件哈希
├─ bundle_info.json
├─ runtime/
│  └─ python/                    # CPython runtime + app-local MSVC CRT/OpenMP
├─ tests/
│  └─ smoke_test.py
└─ results/                       # 唯一的持久写入位置
```

Python 使用 `._pth` 限制模块搜索路径，并以 `-I -s -E -B` 启动。VC++
运行库放在 `python.exe` 同目录，并覆盖 embeddable ZIP 中可能带有的较旧
`vcruntime140*.dll`，确保 Windows DLL 搜索顺序不会先选中旧版。启动器仅修改
子进程环境，不修改系统 `PATH`、注册表、PowerShell 执行策略或用户 Python
配置。`TEMP` / `TMP` 和 `CUDA_CACHE_PATH` 都指向当次运行目录，结束后自动删除。

## 在打包机上生成 ZIP

打包机需要 PowerShell 5.1+ 和可用的 Python + 较新版 pip。pip 只负责
将 CPython 3.11 / Windows x64 wheel 安装到打包目录；打包机自身可以使用其他
Python 版本。准备：

1. 仓库 Release 中的
   `lightgbm-4.7.0.99-1cuda132sm89ptx-win_amd64-cp311-offline.zip`。
2. python.org 官方 `python-3.11.9-embed-amd64.zip`。本次从 python.org 下载后得到的
   SHA-256 为
   `009d6bf7e3b2ddca3d784fa09f90fe54336d5b60f0e0f305c37f400bf83cfd3b`；
   正式发布前还应按 Python 发布页校验 Sigstore 签名。
3. 构建该 DLL 的 Visual Studio 安装中，x64 `Microsoft.VC*.CRT` 和
   `Microsoft.VC*.OpenMP` 目录。应使用与构建工具集一致或更新的可重分发版本。

示例：

```powershell
.\build_portable_bundle.ps1 `
  -WheelhouseZip 'C:\packages\lightgbm-4.7.0.99-1cuda132sm89ptx-win_amd64-cp311-offline.zip' `
  -PythonEmbedZip 'C:\packages\python-3.11.9-embed-amd64.zip' `
  -ExpectedPythonEmbedSha256 '009d6bf7e3b2ddca3d784fa09f90fe54336d5b60f0e0f305c37f400bf83cfd3b' `
  -VcRuntimeDirectories `
    'C:\...\x64\Microsoft.VC145.CRT', `
    'C:\...\x64\Microsoft.VC145.OpenMP' `
  -OutputRoot 'C:\packages\out'
```

打包器会验证当前 Release ZIP 的已知 SHA-256，用 `pip --isolated --no-index`
将 wheel 安装到打包目录，生成包内清单，最后输出一个 ZIP 和该 ZIP 的
SHA-256。打包脚本不覆盖已存在的输出目录或 ZIP。

VC++ DLL 需要符合 Microsoft 的可重分发条款；不要从不明网站收集单个 DLL。

Python 3.11.9 是 Python 3.11 最后一个由 python.org 提供 Windows 二进制的版本；
后续 3.11 安全发布只提供源码。对于不处理不可信输入、用完即删的离线
冒烟测试包，可以暂时保持 CPython 3.11，以严格匹配当前 `cp311`
wheelhouse。如果要长期或对外分发，建议另发一个当前受支持 Python 版本的便携包，
并同步替换 NumPy / SciPy 的 ABI wheel。

## GitHub Actions 发布

[`windows_cuda_portable.yml`](../../.github/workflows/windows_cuda_portable.yml)
会在 Release 发布后自动：

1. 下载该 Release 的 `*cp311-offline.zip` 和 `SHA256SUMS-release.txt`。
2. 下载并校验官方 CPython 3.11.9 embeddable ZIP。
3. 从 GitHub Windows runner 选择 MSVC 14.51 或更新的 x64 CRT/OpenMP
   app-local DLL。
4. 生成 `*-cp311-portable.zip` 及 `.sha256`，上传为 workflow artifact
   并附加到对应 Release。

对已经存在的 Release，在 Actions 中手动运行 **Windows CUDA portable
bundle**，输入 tag，并选择 `publish_to_release=true`。不选择发布时，
产物只作为 workflow artifact 保存。工作流不会覆盖同名 Release 资产。

## 在目标机上测试

1. 把生成的 ZIP 和它的 SHA-256 复制到目标机。
2. 核对 ZIP 哈希，解压到本地目录或可写 U 盘。
3. 双击 `START_TEST.cmd`。
4. 查看 `results\<机器-时间>\result.json` 和 `console.log`。
5. 保存结果后删除整个解压目录即可。

命令行自动化：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\run_portable_test.ps1 -GpuDeviceId 0
```

强制测试 PTX JIT 路径：

```powershell
.\run_portable_test.ps1 -GpuDeviceId 0 -ForcePtxJit
```

## 当前 wheel 的硬件边界

* Windows x64，每次训练一张 NVIDIA GPU。
* Compute Capability 8.9 使用 `sm_89` cubin；理论驱动下限为 R580。
* CC > 8.9 使用 `compute_89` PTX JIT；需要 R595 或更新驱动，且尚未逐卡实测。
* CC < 8.9（例如 RTX 30 / `sm_86`、A100 / `sm_80`）不兼容。“安装了
  CUDA”不能改变这个二进制架构限制。

如果测试机覆盖 RTX 20/30/40/50，应另外构建多架构 wheel，或按 GPU 世代发布
多个便携包；当前 Release 不能用一个 `compute_89` PTX 向下覆盖较旧 GPU。

## “不污染”的精确含义

这套脚本不安装任何系统或用户级组件，应用自身的可控写入都在解压目录内。
但 Windows 事件日志、Defender/EDR 遥测、文件访问日志等由操作系统或企业策略
控制，应用无法保证“系统层面零记录”。
