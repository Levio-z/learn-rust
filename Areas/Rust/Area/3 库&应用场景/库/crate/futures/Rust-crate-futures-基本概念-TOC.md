---
tags:
  - note
---
## 1. 核心观点  
### Ⅰ. 概念层
该箱子提供了若干用于编写异步代码的核心抽象：
- [未来](https://docs.rs/futures/latest/futures/future/index.html "mod futures::future")是异步计算产生的单一最终值。一些编程语言（如 JavaScript）称此为“promise”。
- [流](https://docs.rs/futures/latest/futures/stream/index.html "mod futures::stream")代表一系列异步产生的值。
- [汇（sinks）](https://docs.rs/futures/latest/futures/sink/index.html "mod futures::sink") 支持异步写入数据。
- [执行者](https://docs.rs/futures/latest/futures/executor/index.html "mod futures::executor")负责执行异步任务。

箱子还包含异[步 I/O](https://docs.rs/futures/latest/futures/io/index.html "mod futures::io") 的抽象和 [跨任务沟通](https://docs.rs/futures/latest/futures/channel/index.html "mod futures::channel") 。

这一切的底层是_任务系统_ ，这是一种轻量级线程。大型异步计算通过未来、流和汇构建，然后作为独立任务生成，这些任务被执行到完成， _但不会阻塞_运行线程

以下示例描述了任务系统上下文如何在宏和关键词（如异步和等待！）中构建和使用。

|**子库名称**|**职责**|
|---|---|
|**`futures-core`**|最核心的定义，比如 `Stream` Trait 的基础。|
|**`futures-util`**|**最重要。** 所有的扩展方法（Ext Traits）和大部分逻辑实现。|
|**`futures-sink`**|定义了如何异步地向通道（Channel）发送数据。|
|**`futures-channel`**|提供了异步环境下使用的 `oneshot` 和 `mpsc` 通道。|
|**`futures-io`**|定义了异步读写（`AsyncRead` / `AsyncWrite`）的接口。|
|**`futures-executor`**|提供了一个简单的本地执行器（通常用于测试）。|
### Ⅱ. 应用层




### Ⅲ. 实现层

### **IV**.原理层


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
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
