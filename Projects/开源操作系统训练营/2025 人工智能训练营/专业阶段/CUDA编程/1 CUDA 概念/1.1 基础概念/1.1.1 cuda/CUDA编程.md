CUDA (Compute Unified Device Architecture)，NVIDIA 开发的并行计算平台和编程模型
### 特点
- C/C++ 语法
	- 宿主语言为C/C++，使用范围广，应用领域对口，新语法学 习门槛低
- 与 CPU 端协作
	- CPU 需负责整理程序执行和处理逻辑等
- SIMT 模式
	- SIMT: Single Instruction Multiple Threads 一个指令可以被多个线程同时执行
- 自动调度
	- 根据设定的执行参数自动调度资源（优化）
### 线程层级结构
◆ SIMT → 指挥每个线程 → 需要组织结构和编号 
◆ CUDA 的方式：Grid → Block → Thread

![](asserts/Pasted%20image%2020250812125957.png)
![](asserts/Pasted%20image%2020250812130017.png)
![](asserts/Pasted%20image%2020250812130059.png)
block和thread编号都是从0开始的

◆ 每个线程独一无二的编号/索引（idx）： 
◆ idx = Block ID * Block Size + Thread ID

### CUDA 编译流程
◆ 每个 cu: Host 代码与 Device 代码分离 
◆ 每个虚拟架构：Device 代码编译出 fatbin 
◆ Host 端使用系统的 C++ 编译器（如 g++) 
◆ 链接（device，host) 
◆ 最终获得可使用 GPU 的可执行二进制文 件 
◆ 疑问：架构是指？
![](asserts/Pasted%20image%2020250812133413.png)
https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#compilation-phases

### 英伟达 GPU 架构
◆前面拿 A100 为例，A100 为 Ampere 架构
◆各个架构间有区别 
◆Compute Capability (CC)
	◆类似版本，表示能支持的功能和指令集合
	◆A100 (Ampere 架构) 是 cc 8.0 
◆虽然 A100 举例，但从 CUDA 编程的角度目前各种架构没有本质区别

![](asserts/Pasted%20image%2020250812133731.png)
- https://en.wikipedia.org/wiki/CUDA#GPUs_supported

◆ 具体列表可参考： 
◆ -arch: https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html?highlight=arch#gpu-architecture-arch
◆ -code:https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html?highlight=arch#gpu-code-code-code

![](asserts/Pasted%20image%2020250812134303.png)






### 什么是并行
串行:等待一个任务完成，然后处理其他任务
- 同一个时间点只能处理一个任务
	- 阻塞，第一个任务处理完，处理不了第二个
并行：来回切换不同任务的处理
并行：四个接待员同时处理不同的任务

### 计算机怎么并行
CPU 并行
- 多核，8 核 → 8 个独立的物理核心（同时处理8个不同的任务）
- ...
- 应用：操作系统、Web 服务…
CPU并行
- ???
- 应用：科学计算（模拟/仿真等）、游戏（图形渲 染）、深度学习…

### CPU并行和GPU并行区别？为什么这些选择GPU？
#### CPU vs. GPU
CPU (Central Processing Unit)
- “全能战士”
- 通用和复杂计算/逻辑处理、控制流…
- 少量复杂核心、复杂控制单元、低延迟…
GPU (Graphics Processing Unit)
-  “流水线工人”
- 相对简单的大量数据并行
- 大量简化核心、极简控制单元、高吞吐…

总结：硬件设计上有根本区别，**GPU 专为大规模数据并行设计**，在这些任务下“三个臭皮匠顶个诸葛亮”

### 怎么在GPU上编程？
- 平常写 C++ 程序 → 编译 → 在 CPU 上运行
- 怎么在 英伟达 GPU 上编程呢？
- CUDA (Compute Unified Device Architecture)
	- NVIDIA 开发的并行计算平台和编程模型

![](asserts/Pasted%20image%2020250812111154.png)
- 流多处理器
#### 与CPU协作过程
![](asserts/Pasted%20image%2020250812111246.png)
![](asserts/Pasted%20image%2020250812111529.png)

![](asserts/Pasted%20image%2020250812111704.png)



