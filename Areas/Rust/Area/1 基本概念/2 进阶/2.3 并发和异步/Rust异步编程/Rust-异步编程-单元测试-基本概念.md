---
tags:
  - permanent
---
## 1. 核心观点  
### Ⅰ. 概念层

### Ⅱ. 实现层
你只能在异步上下文中使用 await，而 Rust 中的单元测试并非异步的。大多数运行时都为测试提供了一个便捷属性，类似于 `async main` 的属性。以 Tokio 为例，代码如下所示：
```rust
#[tokio::test]
async fn test_something() {
  // Write a test here, including all the `await`s you like.
}
```
测试的配置方式有很多种，详情请参阅[文档](https://docs.rs/tokio/latest/tokio/attr.test.html) 。

异步代码测试中还有一些更高级的主题（例如，测试竞态条件、死锁等），我们将在本指南的[后面部分](https://rust-lang.github.io/async-book/part-guide/more-async-await.html)介绍其中的一些主题。
### Ⅲ. 原理层



## 2. 背景/出处  
- 来源：
	- [RFC 2394: async/await](https://rust-lang.github.io/rfcs/2394-async_await.html?accessToken=eyJhbGciOiJIUzI1NiIsImtpZCI6ImRlZmF1bHQiLCJ0eXAiOiJKV1QifQ.eyJleHAiOjE3NDg5NjI5MTEsImZpbGVHVUlEIjoiS2xrS3ZlZ1pvZXVkdzdxZCIsImlhdCI6MTc0ODk2MjYxMSwiaXNzIjoidXBsb2FkZXJfYWNjZXNzX3Jlc291cmNlIiwicGFhIjoiYWxsOmFsbDoiLCJ1c2VySWQiOjU5Nzc4NDgzfQ.GX)
	- https://rust-lang.github.io/async-book/part-guide/more-async-await.html
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  



## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
