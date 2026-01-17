---
tags:
  - permanent
---
## 1. 核心观点  

`stream` Trait类似于`future` ，但在完成前可以产生多个值，类似于标准库中的`Iterator`者特征：

- `Stream` trait 是异步流的基础接口，定义了核心方法 `poll_next`，需要实现者提供具体轮询逻辑。
- `StreamExt` 是对 `Stream` trait 的扩展 trait，提供了大量**便捷的、组合式的流操作方法**，比如 `next()`, `filter()`, `map()` 等。
- 以及早起错误退出，`try_map`、`try_filter` 和 `try_fold`。
- 遗憾的是，`for` 循环无法直接用于 `Stream`（流）。但对于命令式风格的代码，**可以使用 `while let` 循环配合 `next`/`try_next` 函数来实现**。


## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  

### 类”for“的遍历手段
```rust
async fn sum_with_next(mut stream: Pin<&mut dyn Stream<Item = i32>>) -> i32 {
    use futures::stream::StreamExt; // for `next`
    let mut sum = 0;
    while let Some(item) = stream.next().await {
        sum += item;
    }
    sum
}

async fn sum_with_try_next(
    mut stream: Pin<&mut dyn Stream<Item = Result<i32, io::Error>>>,
) -> Result<i32, io::Error> {
    use futures::stream::TryStreamExt; // for `try_next`
    let mut sum = 0;
    while let Some(item) = stream.try_next().await? {
        sum += item;
    }
    Ok(sum)
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
 
  
