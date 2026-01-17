---
tags:
  - permanent
---
## 1. 核心观点  

对于返回`结果的`期货，考虑使用 `try_join！` 代替 `join！` 由于 `join！` 只有在所有子未来都完成后才会完成，即使其中一个子未来返回了`error` ，它仍会继续处理其他未来。

与 `join！` 不同，`try_join`！如果其中一个子未来返回错误，会立即完成。

### 整合错误类型
 use futures::future::TryFutureExt;
传递给 `try_join！` 的期货必须都具有相同的错误类型。考虑使用 `.map_err（|e| ...）` 和 `.err_into（）` 函数 `futures：：future：：TryFutureExt` 以整合错误类型：
## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  

```rust
async fn example_try_join() {

    println!("\n=== try_join!使用示例 ===");

    use futures::future::TryFutureExt;

    // 使用try_join!处理Result类型的future

    match try_join!(

        get_book_result(),

        get_music_result().map_err(|e| e.to_string())

    ) {

        Ok((book, music)) => {

            println!("成功获取: {}, {}", book, music);

        }

        Err(e) => {

            println!("获取失败: {}", e);

        }

    }

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
 
  
