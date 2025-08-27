WSL ubuntu 24.04
```
wget https://developer.download.nvidia.com/compute/cuda/13.0.0/local_installers/cuda_13.0.0_580.65.06_linux.runsudo sh cuda_13.0.0_580.65.06_linux.run
```
设置环境变量
```
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
```

### Windows



1. 安装CUDA Toolkit：

首先，你需要从NVIDIA官网下载并安装CUDA Toolkit。确保选择与你的GPU兼容的版本，以及适合Windows操作系统的版本。下载地址：https://developer.nvidia.com/cuda-downloads

https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_network

2. 安装cuDNN（可选）：

如果你打算进行深度学习相关的开发，可能还需要安装cuDNN。同样从NVIDIA官网下载对应版本的cuDNN，并按照说明进行安装。下载地址：https://developer.nvidia.com/rdp/cudnn-archive

3. 安装vs studio 桌面端开发程序并将cl.exe加入环境变量
