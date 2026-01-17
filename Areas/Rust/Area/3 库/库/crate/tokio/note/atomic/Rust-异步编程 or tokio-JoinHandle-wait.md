---
tags:
  - note
---
## 1. 核心观点  
### Ⅰ. 概念层

- **await让主任务会等待生成的任务完成，然后返回结果才**，这称为_连接_任务（类似于[连接](https://doc.rust-lang.org/std/thread/struct.JoinHandle.html#method.join)线程，连接的 API 也类似）。
- **`await` `JoinHandle` 会返回一个 ` `Result**`
- 这就是为什么我们在上面的例子中使用了 `let _ = ...` ，它可以避免关于未使用 `Result` 警告
	- 如果生成的任务成功完成，则任务结果为 ` `Ok` 。
	- 失败见：[Rust-异步编程 or tokio-JoinHandle-wait-panic](Rust-异步编程%20or%20tokio-JoinHandle-wait-panic.md)

### Ⅱ. 应用层


### Ⅲ. 实现层

### **IV**.原理层


## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### 基本例子
- 前置：[Rust-异步编程 or tokio-spawn单独任务运行异步函数](Rust-异步编程%20or%20tokio-spawn单独任务运行异步函数.md)
```rust
use tokio::{spawn, time::{sleep, Duration}};

async fn say_hello() {
    // Wait for a while before printing to make it a more interesting race.
    sleep(Duration::from_millis(100)).await;
    println!("hello");
}

async fn say_world() {
    sleep(Duration::from_millis(100)).await;
    println!("world");
}

#[tokio::main]
async fn main() {
    let handle1 = spawn(say_hello());
    let handle2 = spawn(say_world());
    
    let _ = handle1.await;
    let _ = handle2.await;

    println!("!");
}
```

这次我们不再直接调用 `spawn`，而是保存返回的 `JoinHandle` 并在之后 `await` 它们完成，因此 `main` 中不再需要 `sleep`。

生成的两个任务仍然并发执行，多次运行程序可能看到不同顺序。但由于 `await` 等待了任务完成，最后的感叹号 `'!'` 总是最后打印。

如果立即 `await` 第一个 `spawn`（例如 `spawn(say_hello()).await`），生成的任务会让主任务等待完成才继续执行，完全没有并发，这几乎没有意义——直接顺序调用即可。



## 4. 与其他卡片的关联  
- 前置卡片：
	- [Rust-tokio-基本使用](Rust-tokio-基本使用.md)
	- [Rust-Async和Await-基本概念](../../../../../../1%20基本概念/2%20进阶/2.3%20并发和异步/Async和Await/Rust-Async和Await-基本概念.md)
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
