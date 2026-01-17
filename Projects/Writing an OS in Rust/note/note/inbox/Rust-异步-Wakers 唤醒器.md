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
### 核心思想

waker API 的核心思想是：每次调用 `poll` 时都会传入一个特殊的 `Waker` 类型, 封装在 [`Context`](https://doc.rust-lang.org/nightly/core/task/struct.Context.html) 类型中。这个 `Waker` 类型由执行器创建，可被异步任务用来通知其已完成或者部分完成的状态。因此，执行器无需对之前返回 `Poll::Pending` 的 future 重复调用 `poll` ，直到收到对应 waker 的通知。

通过一个小例子可以很好地说明这一点：
```rust
async fn write_file() {
    async_write_file("foo.txt", "Hello").await;
}
```
此函数会异步地将字符串 “Hello” 写入 `foo.txt` 文件。由于硬盘写入需要一定时间，首次轮询这个 future 时很可能会返回 `Poll::Pending` 。硬盘驱动器会在内部存储传递给 `poll` 调用的 `Waker` ，并在文件写入磁盘时使用它来通知执行器。这样，执行器在收到唤醒通知之前就无需浪费任何时间来尝试轮询该 future。

在这篇文章的实现部分，我们将通过创建一个支持 waker 的自定义执行器来了解 `Waker` 类型的具体工作原理。


## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
