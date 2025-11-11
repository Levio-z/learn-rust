---
tags:
  - permanent
---
## 1. 核心观点  
```
Vec<u8> ──(封装UTF-8约束)──> String ──(Deref视图)──> &str

堆内存: [ h e l l o  ]  
  ↑
 String(vec)  ——Deref→  &str视图
```
也就是说：
- `String` 持有堆上真实数据；
- `&str` 只是一个不可变的“窗口”；
- `Deref` 连接了二者的“使用语义”。
	- 详细见[Rust-自动解引用-Deref语义继承](../../2.1%20所有权、生命周期和内存系统/2.1.3%20生命周期和引用/引用机制/Rust-自动解引用/Rust-自动解引用-Deref语义继承.md)

## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
对于 `String`：

- `impl Deref<Target = str> for String`
- `impl DerefMut for String`
- 可将 `&T` 转为 `&U`，如果 `T: Deref<Target=U>`

> `String` 本质上就是可变的 `str` 包装器，**&String能自动解引用到 `&str`**。



## 4. 与其他卡片的关联  
- 前置卡片：
	- [Rust-str](Rust-str.md)
	- [Rust-String](Rust-String.md)
- 后续卡片：
- 相似主题：
	- [Rust-自动解引用-基本概念-TOC](../../2.1%20所有权、生命周期和内存系统/2.1.3%20生命周期和引用/引用机制/Rust-自动解引用/Rust-自动解引用-基本概念-TOC.md)

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
- [ ] 深入阅读 xxx  
- [ ] 验证这个观点的边界条件  
