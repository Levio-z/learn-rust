
## 1. 核心观点  
```rust
fn new_executor_and_spawner() -> (Executor, Spawner) {

    // Maximum number of tasks to allow queueing in the channel at once.

    // This is just to make `sync_channel` happy, and wouldn't be present in

    // a real executor.

    const MAX_QUEUED_TASKS: usize = 10_000;

    let (task_sender, ready_queue) = sync_channel(MAX_QUEUED_TASKS);

    (Executor { ready_queue }, Spawner { task_sender })

}
```


## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  

这段话解释了在编写简易版执行器（Executor）时，对任务通道容量进行限制的原因。

### 中文翻译

> “允许在通道（Channel）中同时排队任务的最大数量。
> 
> 设置这个值仅仅是为了满足 `sync_channel` 的参数要求（让它能正常工作），在真正的生产级执行器中通常不会存在这种限制。”

---

### 生产级别的执行器
- **动态扩容：** 它们往往使用更复杂的无锁（Lock-free）队列，可以根据负载动态调整，而不是写死一个 `sync_channel` 的容量。
    
- **非阻塞设计：** 生产环境更倾向于使用异步的、非阻塞的任务调度机制，而不是让生产者在通道满了之后原地“死等”。
    
- **侵入式链表：** 许多高性能执行器会将任务直接链接在一起，而不是把它们塞进一个固定长度的数组容器里。

## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
- [x] 深入阅读更多详情请参阅 [_Rust中的零成本 futures_](https://aturon.github.io/blog/2016/08/11/futures/) 文章，它宣布了 futures 被加入 Rust 生态系统的消息。
  


