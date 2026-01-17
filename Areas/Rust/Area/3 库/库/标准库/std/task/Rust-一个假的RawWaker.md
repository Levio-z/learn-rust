---
tags:
  - fleeting
---
## 1. 核心观点  
### Ⅰ. 概念层

RawWaker 允许任务执行器的实现者创建 Waker或者使用提供自定义唤醒行为的 LocalWaker 器。

它由数据指针和[虚函数指针表（vtable）](https://en.wikipedia.org/wiki/Virtual_method_table) 组成。 自定义 `RawWaker` 的行为。

使用 `RawWaker` 是不安全的。实现 [`Wake`](https://doc.rust-lang.org/alloc/task/trait.Wake.html) trait 是一种安全的替代方案，但需要分配内存。

`RawWaker` 是 Rust 异步运行时底层用于构造 `Waker` 的原始类型，其本质是一个 **手动实现运行时多态（runtime polymorphism）** 的结构。


它由两部分组成：

1. `data: *const ()` —— 一个类型擦除（type-erased）的数据指针
    
2. `vtable: &'static RawWakerVTable` —— 一个显式定义的“虚方法表”
	- 该表指定了当 RawWaker 被克隆、唤醒或被释放时应当调用的函数。

`RawWaker` 通过显式定义的 `RawWakerVTable`，手动实现了一套运行时多态机制。  
它利用 `*const ()` 进行类型擦除，使非泛型的 `RawWaker` 能承载任意任务状态；同时也将 **类型安全、生命周期管理、线程安全** 的全部责任交还给程序员。这种设计是为了满足异步运行时在零抽象成本、跨 crate、跨平台场景下的极致需求，但其危险性极高，因此只适合底层框架作者使用。
### Ⅱ. 应用层




### Ⅲ. 实现层

### **IV**.原理层


## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  

```rust
// in src/task/simple_executor.rs

use core::task::RawWakerVTable;

fn dummy_raw_waker() -> RawWaker {
    fn no_op(_: *const ()) {}
    fn clone(_: *const ()) -> RawWaker {
        dummy_raw_waker()
    }

    let vtable = &RawWakerVTable::new(clone, no_op, no_op, no_op);
    RawWaker::new(0 as *const (), vtable)
}
```

- 首先，我们定义两个名为 `no_op` 和 `clone` 的内部函数。`no_op` 函数接收一个 `*const ()` 指针且不执行任何操作。 `clone` 函数同样接收一个 `*const ()` 指针并通过再次调用 `dummy_raw_waker` 返回一个新的 `RawWaker`。我们使用这两个函数来创建一个最简的 `RawWakerVTable`：`clone` 函数用于克隆操作，而 `no_op` 函数则用于所有其他操作。由于这个 `RawWaker` 不做任何实际工作，因此从 `clone` 返回一个新的 `RawWaker` 而非克隆它本身也没关系。

- 创建完 `vtable` 后，我们使用 `RawWaker::new` 函数来创建 `RawWaker`。被传递的 `*const ()` 无关紧要，因为 vtable 中没有任何一个函数使用它。因此，我们只需传递一个空指针。

####   一个 `run` 方法
既然我们已经掌握了创建 `Waker` 实例的方法，就可以用它为我们的执行器实现一个 `run` 方法。 最简单的 `run` 方法就是在循环中不断轮询所有排队中的任务，直到它们全部完成。这种方式效率不高，因为它没有利用 `Waker` 类型的通知机制，但这是一个快速上手的简单方法：
```rust
```rust
// in src/task/simple_executor.rs

use core::task::{Context, Poll};

impl SimpleExecutor {
    pub fn run(&mut self) {
        while let Some(mut task) = self.task_queue.pop_front() {
            let waker = dummy_waker();
            let mut context = Context::from_waker(&waker);
            match task.poll(&mut context) {
                Poll::Ready(()) => {} // 任务完成
                Poll::Pending => self.task_queue.push_back(task),
            }
        }
    }
}
```




## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：[Rust-RawWaker](Rust-RawWaker.md)
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
