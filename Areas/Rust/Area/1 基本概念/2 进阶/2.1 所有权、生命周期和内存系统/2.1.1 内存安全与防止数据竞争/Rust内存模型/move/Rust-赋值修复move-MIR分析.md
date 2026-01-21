---
tags:
  - note
---
## 1. 核心观点  
### Ⅰ. 概念层

**赋值操作改变对象的地址**：
- Rust 中的**赋值操作会移动数据**（大多数情况下是这样，一些简单的数据具有复制语义，但这并不重要）。当我们写 `let b = a;` 时，内存中位于 `a` 地址的数据会被移动到 `b` 地址。这意味着赋值之后，数据存在于 `b` ，但不再存在于 `a` 。换句话说， [赋值](https://rust-lang.github.io/async-book/part-reference/pinning.html#footnote-compiler)操作改变了对象的地址。
- 如果之前move还会撤销move状态，被赋值的变量可以重新使用。见[MIR角度解释](#MIR角度解释)

### Ⅱ. 实现层



### Ⅲ. 原理层

## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### MIR角度解释
初始设置：为什么赋值能“修复” move 后状态
```
n.next = new_value;
```
MIR 中等价于：
```rust
write n.next = new_value
drop_flag(n.next) = true
```
于是：
```rust
n.val  : initialized
n.next : initialized
```

不是撤销 move，而是写入了一个全新的合法值
### 与 `Option::take()` 的统一解释
```rust
let old = n.next.take();
```
MIR 等价于：
```
tmp = move n.next
n.next = None
```
也就是：
- move
- 立即回填
- 从未进入“部分未初始化”的危险状态




## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
