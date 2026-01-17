---
tags:
  - note
---
## 1. 核心观点  
### Ⅰ. 概念层

它是一个通用运行时，也是该生态系统中最流行的运行时。对于入门和生产环境来说，它都是一个不错的选择。

### Ⅱ. 应用层

### 多线程运行时
- Tokio（在其默认配置下）是一个多线程运行时，这意味着当我们生成一个新任务时，该任务可能运行在与它所源自的任务不同的操作系统线程上（它可能运行在同一个线程上，或者它可能在一个线程上启动，然后稍后被移到另一个线程）。

- 因此，当一个 Future 被创建为一个任务时，它会与创建它的任务以及其他任何任务_并发_运行。如果它被调度到不同的线程上，它也可能与这些任务并行运行。

|**报错的代码片段 (API)**|**提示错误原因**|**必须补充的 Feature**|
|---|---|---|
|`tokio::spawn` 或 `#[tokio::main]`|`cannot find function...`|`rt` 或 `rt-multi-thread`|
|`tokio::time::sleep`|`module 'time' not found`|**`time`**|
|`tokio::fs::File`|`module 'fs' not found`|**`fs`**|
|`AsyncReadExt`/`AsyncWriteExt`|无法调用 `.read_exact()` 等方法|**`io-util`**|
|`tokio::sync::mpsc`|`module 'sync' not found`|**`sync`**|
|`tokio::process::Command`|`module 'process' not found`|**`process`**|

### Ⅲ. 实现层

### **IV**.原理层
### 核心逻辑
![](asserts/Pasted%20image%2020260105184851.png)
### 异步运行如何做的
![](asserts/Pasted%20image%2020260105185414.png)
### 为什么需要pin
主要针对栈上的内存，主要是move是memcpy
![](asserts/Pasted%20image%2020260105191821.png)




## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
详细阐述这个观点，包括逻辑、例子、类比。  
- 要点1  
- 要点2  

## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
	- [Rust-tokio-基本使用](Rust-tokio-基本使用.md)
	- [Rust-异步编程 or tokio-JoinHandle](Rust-异步编程%20or%20tokio-JoinHandle.md)
	- [Rust-异步编程 or tokio-JoinHandle-wait](Rust-异步编程%20or%20tokio-JoinHandle-wait.md)
	- [Rust-异步编程 or tokio-JoinHandle-abort](Rust-异步编程%20or%20tokio-JoinHandle-abort.md)
	- [Rust-异步编程 or tokio-JoinHandle-wait-panic](Rust-异步编程%20or%20tokio-JoinHandle-wait-panic.md)
	- [Rust-tokio-生态](Rust-tokio-生态.md)
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
