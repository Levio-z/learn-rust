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

![image.png](https://raw.githubusercontent.com/Levio-z/MyPicture/main/img/20260116160350805.png)

- prost：Protobuf处理
- tokio-utils：实用工具（如Framed/Codec）
- tokio-stream：流处理工具
- loom：并发测试工具
### Tokio文档

- 核心功能:
    - 异步运行时：提供多线程运行时执行异步代码
    - 异步标准库：提供异步版本的std功能
    - 丰富生态：包含大量实用库和工具
- 适用场景:
    - IO密集型：网络应用、文件读写等
    - 不适用场景:
        - CPU密集型计算（推荐使用rayon）
        - 大量文件读取（操作系统不支持异步文件API）
        - 单次网络请求（使用阻塞API更简单）
- 独特优势:
    - 内存安全：防止常见错误如无界队列、缓冲区溢出
    - 高性能调度：基于工作窃取的多线程调度器
    - 灵活配置：可适应从大型服务器到嵌入式设备
    - 完整生态：包含生产环境所需全部组件
### API
- 核心模块:
    - task：轻量级非阻塞执行单元
        - spawn：创建新任务
        - JoinHandle：等待任务完成
    - sync：同步原语
        - 通道（oneshot、mpsc、broadcast等）
        - 异步Mutex
        - Barrier同步
    - time：时间跟踪和调度
    - runtime：运行时配置和管理
- 特性选择:
    - rt：基本运行时功能
    - rt-multi-thread：多线程工作窃取调度器
    - net：TCP/UDP支持
    - macros：#[tokio::main]等宏
    - 最佳实践：库作者应仅启用所需特性
- 开发模式:
    - 使用#[tokio::main]宏启动运行时
    - CPU密集型任务应使用spawn_blocking
    - **注意避免长时间不await的代码块**
	    - 标准库Mutex不能跨越await点




## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
