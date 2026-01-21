---
tags:
  - note
---
## 1. 核心观点  
### Ⅰ. 概念层

它是一个通用运行时，也是该生态系统中最流行的运行时。对于入门和生产环境来说，它都是一个不错的选择。Tokio是Rust异步运行时生态系统的核心组件，为构建高性能网络应用提供基础架构。是Rust开发者必须掌握的核心工具，尤其在需要高并发的网络编程场景。

### 基础架构
- 核心功能:
    - 异步运行时：提供[多线程运行时](#多线程运行时)执行异步代码
    - 异步标准库：提供异步版本的std功能
    - 丰富生态：包含大量实用库和工具
- 适用场景:
    - IO密集型：网络应用、文件读写等
    - 不适用场景:
        - CPU密集型计算（推荐使用rayon）
        - 大量文件读取（操作系统不支持异步文件API）
        - 单次网络请求（使用阻塞API更简单）
### 核心模块:
- task：轻量级非阻塞执行单元
	- spawn：创建新任务
	- JoinHandle：等待任务完成
- sync：同步原语
	- 通道（oneshot、mpsc、broadcast等）
	- 异步Mutex
	- Barrier同步
- time：时间跟踪和调度,高性能定时器: 精确控制异步任务执行时机
- runtime：运行时配置和管理,提供任务调度和执行环境
- I/O驱动: 基于操作系统事件队列（epoll/kqueue/IOCP）
### 特性支持
- [tokio特性](#tokio特性)
### Ⅱ. 应用层
- [谁在使用tokio](#谁在使用tokio)




### Ⅲ. 实现层

### **IV**.原理层
### 核心逻辑
![](../../tokio/asserts/Pasted%20image%2020260105184851.png)
### 异步运行如何做的
![](../../tokio/asserts/Pasted%20image%2020260105185414.png)
### 为什么需要pin
主要针对栈上的内存，主要是move是memcpy
![](../../tokio/asserts/Pasted%20image%2020260105191821.png)




## 2. 背景/出处  
- 来源：
	- 
- 引文/摘要：  
  - …  
  - …  
- 参考文档：
	- https://tokio.rs/tokio
## 3. 展开说明  


- 独特优势:
    - 内存安全：防止常见错误如无界队列、缓冲区溢出
    - 高性能调度：基于工作窃取的多线程调度器
    - 灵活配置：可适应从大型服务器到嵌入式设备
    - 完整生态：包含生产环境所需全部组件
### 多线程运行时
- Tokio（在其默认配置下）是一个多线程运行时，这意味着当我们生成一个新任务时，该任务可能运行在与它所源自的任务不同的操作系统线程上（它可能运行在同一个线程上，或者它可能在一个线程上启动，然后稍后被移到另一个线程）。

- 因此，当一个 Future 被创建为一个任务时，它会与创建它的任务以及其他任何任务_并发_运行。如果它被调度到不同的线程上，它也可能与这些任务并行运行。
### API


- 开发模式:
    - 使用#[tokio::main]宏启动运行时
    - CPU密集型任务应使用spawn_blocking
    - **注意避免长时间不await的代码块**
	    - 标准库Mutex不能跨越await点

### tokio特性

>初学者可使用full特性启用全部功能，但会引入较多依赖


- 特性选择:
	- ==常用特性
	    - rt：基本运行时功能，包含单线程调度器
	    - rt-multi-thread：多线程工作窃取调度器
	    - net：TCP/UDP支持
	    - macros：#[tokio::main]等宏
    - 最佳实践：库作者应仅启用所需特性
    - 其他特性
	    - fs: 异步文件系统操作

#### 建议
- 库开发: **只启用必要特性以减少依赖**

| **报错的代码片段 (API)**                 | **提示错误原因**                   | **必须补充的 Feature**        |
| --------------------------------- | ---------------------------- | ------------------------ |
| `tokio::spawn` 或 `#[tokio::main]` | `cannot find function...`    | `rt` 或 `rt-multi-thread` |
| `tokio::time::sleep`              | `module 'time' not found`    | **`time`**               |
| `tokio::fs::File`                 | `module 'fs' not found`      | **`fs`**                 |
| `AsyncReadExt`/`AsyncWriteExt`    | 无法调用 `.read_exact()` 等方法     | **`io-util`**            |
| `tokio::sync::mpsc`               | `module 'sync' not found`    | **`sync`**               |
| `tokio::process::Command`         | `module 'process' not found` | **`process`**            |
### 谁在使用tokio
- 由AWS、Discord、Azure等知名公司支持，Cloudflare的Pingora负载均衡器即基于Tokio构建
## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
	- [Rust-tokio-基本使用](Rust-tokio-基本使用.md)
	- [Rust-异步编程 or tokio-JoinHandle-TOC](Rust-异步编程%20or%20tokio-JoinHandle-TOC/Rust-异步编程%20or%20tokio-JoinHandle-TOC.md)
	- [Rust-异步编程 or tokio-JoinHandle-wait](Rust-异步编程%20or%20tokio-JoinHandle-TOC/Rust-异步编程%20or%20tokio-JoinHandle-wait.md)
	- [Rust-异步编程 or tokio-JoinHandle-abort](Rust-异步编程%20or%20tokio-JoinHandle-TOC/Rust-异步编程%20or%20tokio-JoinHandle-abort.md)
	- [Rust-异步编程 or tokio-JoinHandle-wait-panic](Rust-异步编程%20or%20tokio-JoinHandle-TOC/Rust-异步编程%20or%20tokio-JoinHandle-wait-panic.md)
	- [Rust-tokio-生态](Rust-tokio-生态.md)
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
