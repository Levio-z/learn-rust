---
tags:
  - permanent
---
## 1. 核心观点  

宏允许在同时执行多个不同未来时等待它们完成。

>“`join!` 宏类似于 `.await`，但它可以**同时等待多个 Future 运行**。
> 
> 如果我们暂时在 `learn_and_sing`（学习并唱歌）这个 Future 中被阻塞了（例如在等待下载乐谱），那么 `dance`（跳舞）这个 Future 将会**接管当前线程**并运行。
> 
> 如果 `dance` 随后也被阻塞了，`learn_and_sing` 又可以**重新夺回控制权**。
> 
> 如果两个 Future 同时都处于阻塞状态，那么整个 `async_main` 就会进入阻塞并**主动让出（Yield）** CPU 资源给执行器（Executor）。”

## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
```rust
async fn learn_and_sing() {
    // Wait until the song has been learned before singing it.
    // We use `.await` here rather than `block_on` to prevent blocking the
    // thread, which makes it possible to `dance` at the same time.
    let song = learn_song().await;
    sing_song(song).await;
}

async fn async_main() {
    let f1 = learn_and_sing();
    let f2 = dance();

    // `join!` is like `.await` but can wait for multiple futures concurrently.
    // If we're temporarily blocked in the `learn_and_sing` future, the `dance`
    // future will take over the current thread. If `dance` becomes blocked,
    // `learn_and_sing` can take back over. If both futures are blocked, then
    // `async_main` is blocked and will yield to the executor.
    futures::join!(f1, f2);
}

fn main() {
    block_on(async_main());
}
```

## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
