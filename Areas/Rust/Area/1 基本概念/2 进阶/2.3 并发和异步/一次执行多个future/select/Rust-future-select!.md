---
tags:
  - permanent
---
## 1. 核心观点  

`futures：：select` 宏可以同时运行多个未来，允许用户在任何 future 完成后立即响应。

-  complete => break,
	- 分支可用于处理所有被`选`中的未来都已完成且不再有进展的情况。This is often handy when looping over a `select!`.
-  default => unreachable!(), // never runs (futures are ready, then complete)
	- 如果被`select`的未来尚未完成，默认分支将运行。因此，带有`default`分支的`select`总是立即返回 ，他未来都还没准备好，默认就会被执行

`select`中使用的未来必须实现两个 `Unpin` 以及`（FusedFuture`）特性。[Rust-future-fuse](../fuse/Rust-future-fuse.md)
## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
```rust
use futures::{future, select};

async fn count() {
    let mut a_fut = future::ready(4);
    let mut b_fut = future::ready(6);
    let mut total = 0;

    loop {
        select! {
            a = a_fut => total += a,
            b = b_fut => total += b,
            complete => break,
            default => unreachable!(), // never runs (futures are ready, then complete)
        };
    }
    assert_eq!(total, 10);
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
 
  
