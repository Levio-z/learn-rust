---
tags:
  - note
---
## 1. 核心观点  
### Ⅰ. 概念层
当然有！Rust 为所有的基础整数类型（如 `u8`, `u32`, `i64`, `usize` 等）都提供了一整套**饱和运算（Saturating Arithmetic）**家族。

- [1. saturatingadd (加法)](#1.%20saturatingadd%20(加法))
- [2. saturatingmul (乘法)](#2.%20saturatingmul%20(乘法))
- [3. saturatingpow (幂运算)](#3.%20saturatingpow%20(幂运算))
- [4. saturatingabs (绝对值)](#4.%20saturatingabs%20(绝对值))

### Ⅱ. 应用层

### 什么情况用饱和
当你处理**用户可见的数值**或者**索引/游标**时，一定要显式使用饱和运算：

- **进度条**：进度不能超过 100%。
    
- **游标移动**：就像你之前代码里的 `self.cur.saturating_add(steps)`，游标不能移出数据的最大长度，否则会导致后续读取时发生数组越界。
    
- **游戏数值**：经验值满了就停在等级上限，而不是突然变成 0 级。

### 为什么饱和加法 `saturating_add` 不是默认的？

1. **性能开销**：饱和加法需要 CPU 在运算后多做一次比较（检查结果是否超过边界并赋值）。虽然这很微小，但在底层高性能计算中累积起来也是一笔开销。
    
2. **隐藏 Bug**：饱和运算会“悄悄地”修正你的结果。有时候你并不希望数字卡在 `255`，而是希望程序告诉你“数字出错了”。强制 Panic（Debug 模式）能帮你更早发现逻辑漏洞。


### Ⅲ. 实现层

### **IV**.原理层


## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  


除了你已经用过的 `saturating_sub`（减法），还有以下成员：

### 1. `saturating_add` (加法)

当结果超过该类型的最大值（`MAX`）时，它会停在 `MAX`。
- **场景**：计算游戏角色的经验值或血量。

```
let health: u8 = 240;
let heal_amount: u8 = 30;
// 240 + 30 = 270，超过了 u8 的 255
let final_health = health.saturating_add(heal_amount);
assert_eq!(final_health, 255); 
```

### 2. `saturating_mul` (乘法)

当乘法结果溢出时，返回最大值或最小值。

- **场景**：计算总价。如果因为数量巨大导致总价溢出，返回最大可能值，而不是变成一个奇怪的负数或小数字。

```
let price: u32 = 1_000_000;
let quantity: u32 = 5000;
// 结果远超 u32 最大值 (~42亿)
let total = price.saturating_mul(quantity);
assert_eq!(total, u32::MAX);
```

### 3. `saturating_pow` (幂运算)

计算 ![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB4AAAAlCAYAAABVjVnMAAACzklEQVR4AeyVS6hOURTHP29Skjd5Z0CIvENkwgCRAQMTUZQBA8qEGSkxkSQmHsnIwGuCSN6PvAoZoOT9JiSve3+/c7997rnde+vec869d3K+/r+z1t7fOXvts/ba+7QutdCvCNxsiS9SXaS6yTKQV3G1YYbTYTzot8IOhoUwAuzDVCuvwKsZcjYcgk2wC1ZAFzgB28DJYKqUR+DeDDUMDPYOuxYOgBM4iL0MZqIzNlYegQcw2jVoD05gL/YmqI5cesBfqIBYeQS+xWhHYDj0gjMQgvTFHwPX4QfEyiNwGGwSzkd4CkGTcbrBSaihugK3444JMB+6Q5D9M2hYRDXWiz5TOg37AN6CastlAdyHh7AcZkKkZGCrbh69b2A/bIX3sA5Gwh3YWeYZ1hRiIrmOtpMp7ck/U+E4dAD9u9hIycCL6LHs3Y+j8EeD1WnfWfwtMAt82EEX4wdZYGbB9Q19H3CsaMc9jL8HvkKkELg/rQ2wCh6BskBe4XjPY6z7cQp2CahvXsrcwA6FCxD0B2cpzAGXzSLErZKD6s3l8hyuQpDrNrbccObf8Z3IS+xF8C0wkf5x/QxOFhPLtgX3K+4pOyGwgyyjz/2GiWRhubY+fD7qKZXuYc2ORfYCP7VCYN9GkgN5xg6kwypNbhG6sisErmskjzkr/TZ/mmJMfqovsOtrOo1kFSfXyC+N28d97f+pCIGt5t+McBQ6QT8YB+qSlwSeUOdoezxi0snAzn4Nj/sGnrfuRwfvQ99P8IuDieQ9K/GOQab0G9jUmVpPqfUM+B/8vro9fHsPC7pK+htxrOrtWKsdk04G9o1283hX2AdP4DS4h03pKXy3jtVtpXuAxCcQ/6WSgZ35Dp72KzIROwQ2gweKp84gfI9Ht5b2E+3MMnAYxH38msYXCPJE8m2v0JHsp5lNycDZRmrk00XgRiasobfXvq9Ide2cNFFPJQAAAP//L/qa4QAAAAZJREFUAwD5r3xL/ayG/AAAAABJRU5ErkJggg==)。幂运算增长极快，非常容易溢出。

```
let base: u32 = 2;
// 2 的 32 次方超出了 u32 的范围
let result = base.saturating_pow(32);
assert_eq!(result, u32::MAX);
```

### 4. `saturating_abs` (绝对值)

这是最特殊的一个，仅针对**有符号整数**（如 `i8`, `i32`）。

- **原理**：在计算机中，负数比正数“多一个”。比如 `i8` 的范围是 `-128` 到 `127`。
    
- **痛点**：如果你求 `-128` 的绝对值，结果应该是 `128`，但 `i8` 装不下 `128`！
    
- **结果**：调用 `(-128i8).saturating_abs()` 会得到 `127`。
    

---

### 总结：什么时候用饱和运算？

你可以把这些方法想象成给数值运算装上了“安全护栏”：

|**方法**|**核心逻辑**|**生活类比**|
|---|---|---|
|**`saturating_add`**|向上封顶|往杯子里倒水，满了就不再增加。|
|**`saturating_sub`**|向下保底|电梯下降，降到 1 楼（或 B1）就到底了。|
|**`saturating_mul`**|规模控制|无论怎么翻倍，最终不超过容器上限。|

### 进阶：如何处理“负数”的饱和？

对于有符号整数（如 `i32`），饱和运算会同时处理两个边界：

- 如果向负方向溢出（比如 `-21亿` 减去 `10亿`），它会停在 `i32::MIN`（约 `-21.4亿`）。
    

**既然你掌握了这些安全方法，下次在做 LeetCode 的计算类题目（比如计算平方根、反转整数、计算总数）时，试着用这些方法代替普通的 `+ - *`，你会发现能省掉大量的 `if (overflow) ...` 判断。**

你想尝试用 `saturating_add` 来写一个简单的“计数器”组件吗？我们可以确保它永远不会因为点击太快而崩溃。

## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
	- [英语-单词-saturae](../../../../../../../basic/英语/Areas/单词/s/英语-单词-saturae.md)
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
- [x] 深入阅读 xxx
- [x] 验证这个观点的边界条件
