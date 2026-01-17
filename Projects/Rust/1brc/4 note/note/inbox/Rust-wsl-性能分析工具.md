---
tags:
  - note
---
## 1. 核心观点  
### Ⅰ. 概念层



### Ⅱ. 应用层




### Ⅲ. 实现层

### **IV**.原理层


## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
在 WSL (Ubuntu/Linux) 环境下进行 Rust 代码性能分析，主要分为**采样（Sampling）**和**可视化（Visualization）**两个阶段。

针对你提到的“查看具体方法”和“堆栈占用时间”，以下是目前 Rust 生态中最主流、最强大的工具组合：

---

## 1. 核心推荐：`cargo-flamegraph`

这是 Rust 开发者最常用的工具，它基于底层的 `perf` 工具，能够生成火焰图（Flamegraph）。

- **作用**：直观展示哪个函数（方法）占用了最多的 CPU 时间，堆栈深度一目了然。
    
- **安装**：
    
    Bash
    
    ```
    # 安装底层依赖
    sudo apt update && sudo apt install linux-tools-common linux-tools-generic -y
    # 安装工具
    cargo install cargo-flamegraph
    ```
    
- **使用**：
    
    Bash
    
    ```
    cargo flamegraph
    ```
    
- **分析方法**：生成的 `flamegraph.svg` 中，**横轴越宽**代表该函数占用 CPU 时间比例越高；**纵轴**代表调用堆栈。
    

---

## 2. 精准堆栈分析：`samply`

如果你觉得火焰图还不够细致，`samply` 是目前极其推荐的新兴工具（由 Mozilla 开发）。它在 WSL 下表现极佳，提供了一个强大的 Web 界面。

- **安装**：
    
    Bash
    
    ```
    cargo install samply
    ```
    
- **使用**：
    
    Bash
    
    ```
    samply record cargo run --release
    ```
    
- **特点**：
    
    - **时间线视图**：可以看到程序在运行过程中的每一个毫秒都在干什么。
        
    - **堆栈追踪**：点击具体方法，可以直接看到完整的调用栈及各级耗时百分比。
        
    - **零配置**：不需要手动配置特殊的符号表，它处理得非常好。
        

```
'/proc/sys/kernel/perf_event_paranoid' is currently set to 2.
In order for samply to work with a non-root user, this level needs
to be set to 1 or lower.
You can execute the following command and then try again:
    echo '1' | sudo tee /proc/sys/kernel/perf_event_paranoid
```

- [linux-配置-perf_event_paranoid](linux-配置-perf_event_paranoid.md)
---

## 3. 实时性能监控：`bottom` 或 `htop`

如果你想在代码运行时实时观察系统资源的波动（虽然不针对特定方法，但对宏观分析很有用）：

- **工具**：`btm` (Rust 编写)
    
- **安装**：`cargo install bottom`
    

---

## 4. 针对 WSL 的特殊配置（必做）

在 WSL 中使用 `perf` 或 `cargo-flamegraph` 时，通常会遇到权限限制，导致无法读取内核指针或采样。

**必须运行以下命令来放开权限：**

Bash

```
# 允许非 root 用户进行采样
sudo sysctl -w kernel.perf_event_paranoid=1
# 允许查看内核符号地址
sudo sysctl -w kernel.kptr_restrict=0
```

_注：这些设置在重启后会失效，建议加入到 `~/.bashrc` 中。_

---

## 5. 编译优化建议

为了让工具能准确识别“具体方法名”，你需要确保二进制文件中包含**调试符号**，同时开启**优化**。

在你的 `Cargo.toml` 中添加：

Ini, TOML

```
[profile.release]
debug = true  # 在 release 模式下保留调试符号，以便分析工具读取方法名
```

---

### 原子对比

|**工具**|**优势**|**适合场景**|
|---|---|---|
|**cargo-flamegraph**|产出直观、行业标准|寻找性能瓶颈“大头”|
|**samply**|交互式界面、精细到毫秒|深度堆栈分析、解决长尾延迟|
|**perf**|原生、功能最全|极高阶用户，分析系统调用|

---

**你想先从生成一张简单的火焰图开始，还是需要我帮你分析一段特定的 Rust 代码该如何打点（Instrumentation）？**




## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
