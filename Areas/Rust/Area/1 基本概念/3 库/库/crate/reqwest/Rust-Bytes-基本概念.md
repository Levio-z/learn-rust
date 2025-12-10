---
tags:
  - note
---
## 1. 核心观点  
### Ⅰ. 概念层

在 Rust 的 `reqwest` 或 `bytes` crate 中，`Bytes` 是一个高效的 **不可变字节缓冲区类型**，用于处理 HTTP body 或二进制数据。它主要特点是 **零拷贝、共享引用**，适合高性能场景。

### Ⅱ. 实现层


### Ⅲ. 原理层

## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### 1. 基本定义

来自 `bytes` crate：
```
pub struct Bytes {
    inner: Arc<[u8]>, // 内部用 Arc 管理内存
}
```
- 内部数据通过 `Arc` 管理，可在多个 `Bytes` 对象间共享。
- 不可变：不能直接修改其中的字节。
### 2. 常用方法

| 方法                    | 说明                            |                                                                                             |
| --------------------- | ----------------------------- | ------------------------------------------------------------------------------------------- |
| `len()`               | 返回字节长度                        |                                                                                             |
| `is_empty()`          | 判断是否为空                        |                                                                                             |
| `as_ref()`            | 返回 `&[u8]` 切片                 |                                                                                             |
| `to_vec()`            | 将 `Bytes` 转成 `Vec<u8>`（会拷贝数据） |                                                                                             |
| `slice(start..end)`   | 生成共享的子切片，不拷贝数据                | 根据给定的范围 `begin..end`，返回一个新的 `Bytes` 对象，该对象只包含原 `Bytes` 中指定的部分。仅仅是增加引用计数和记录新的偏移范围。不会遍历或复制数据。 |
| `copy_to_slice(dest)` | 将数据拷贝到已有缓冲区                   |                                                                                             |
### 3 特点
#### 3.1 和很容易转换成`&[u8]`

- `AsRef<[u8]>` 和 `deref到[u8]`

```
 trait：impl AsRef<[u8]> for Bytes
 impl Deref for Bytes type Target = [u8];
```

#### 3.1.1 Deref 实例
因为 `Deref` 指向 `[u8]`，所以 `*tmp` 被解引用成 `[u8]`，再取全切片 `[..]` 得到 `&[u8]`。
```
let x: &[u8] = &resp.bytes().await?[..];
```

## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：
	- 【net】[reqwest 存储库示例](https://github.com/seanmonstar/reqwest/tree/master/examples)
	- 【net】[The Rust Cookbook  Rust 食谱](https://rust-lang-nursery.github.io/rust-cookbook/web/clients.html)

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
- [ ] 深入阅读 xxx  
- [ ] 验证这个观点的边界条件  
