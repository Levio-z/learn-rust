---
tags:
  - permanent
---
## 1. 核心观点  

**你的异步函数产生的任务能否在线程间传递，取决于你在执行 `.await`（函数暂停）的那一刻，手里有没有拽着某些“不能跨线程”的东西（如 `Rc` 或 `std MutexGuard`）。**


当你编写一个 `async fn` 时，编译器会把它转换成一个**状态机（State Machine）**。这个状态机其实就是一个隐藏的结构体，它存储了函数中所有**跨越 `.await` 存活的变量**。

“跨越 `.await` 点持有”是指：一个变量在 `.await` 之前被创建，并且在 `.await` 之后还要被使用。

## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  

"Held across an .await point"？

- **跨越了：**
    
    
    
    ```Rust
    async fn example() {
        let rc = Rc::new(5); // Rc 是 non-Send 的
        do_something().await; // 暂停点
        println!("{}", rc);   // .await 之后还要用，所以 rc 必须保存在状态机里
    } // 结果：这个 Future 不是 Send
    ```
    
- **没跨越：**


    ```Rust
    async fn example() {
        {
            let rc = Rc::new(5);
            println!("{}", rc);
        } // rc 在这里被销毁了
        do_something().await; // 暂停点
    } // 结果：这个 Future 是 Send
    ```
    

---

### 3. 为什么 `Send` 属性如此重要？

在像 `async-std` 或 `tokio` 这样的多线程执行器中，任务会被分发到不同的线程执行。

- 如果一个 `Future` 是 `Send`，执行器可以把它从线程 A 移动到线程 B 运行。
    
- 如果不是 `Send`，执行器就不能跨线程调度它。
    

**判定标准如下表：**

|**变量类型**|**是否跨越 .await**|**Future 是否为 Send**|**结论**|
|---|---|---|---|
|`i32`, `String` (Send)|是|**Yes**|没问题，正常多线程调度。|
|`Rc`, `MutexGuard` (non-Send)|**否**|**Yes**|只要在暂停前销毁就没问题。|
|`Rc`, `MutexGuard` (non-Send)|**是**|**No**|**报错！** 不能在多线程执行器中 spawn。|

---

### 4. 常见的“元凶”：`MutexGuard`

你之前遇到的 `TcpStream` 写入问题，如果涉及到锁，最容易碰到这个报错。

普通的 `std::sync::Mutex` 的锁是不支持跨越 `.await` 的。如果你在持有锁的时候调用了 `.await`，编译器会发现 `MutexGuard`（它是 non-Send 的）被保存在了状态机里，从而导致整个 `Future` 变成 non-Send。




## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
	- [Rust-future-fuse-terminated](Rust-future-fuse-terminated.md)
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
