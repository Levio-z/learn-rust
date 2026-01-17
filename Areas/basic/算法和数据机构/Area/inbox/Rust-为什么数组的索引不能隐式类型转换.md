---
tags:
  - permanent
---
## 1. 核心观点  

- Rust 避免 **隐式类型转换**，因为可能导致溢出或负数索引访问内存
    
- 如果自动把 `i32` 转成 `usize`：
    
    - 负数会 wrap-around → 可能访问 `usize::MAX`，造成 UB
        
- 因此，索引必须显式转换：
- **但是声明数组长度的时候可以**

## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  

明白了，你问的是这个场景：

```rust
let mut ans = vec![vec![0; n]; n];
ans[x as usize][y as usize] = val;
```

为什么 `x`、`y` 不能直接用 `i32` 索引，而必须 `as usize`。我们来逐条分析。

---

### 1. Rust 数组 / Vec 索引规则

在 Rust 中：

- 数组/向量索引类型必须是 **`usize`**
    
- 这是 Rust 设计的一部分，保证内存安全：
    
    - 索引值对应实际内存偏移量
        
    - `usize` 在不同架构下（32/64 位）自动匹配指针大小
        
- 编译器**不会自动将整数类型转换为 `usize`**，即便是 `i32`、`i64` 也不行
    

```rust
let v = vec![10, 20, 30];
let i: i32 = 1;
v[i]; // ❌ 编译错误：expected usize, found i32
```

---

### 2. 为什么不能自动类型推导

- Rust 避免 **隐式类型转换**，因为可能导致溢出或负数索引访问内存
    
- 如果自动把 `i32` 转成 `usize`：
    
    - 负数会 wrap-around → 可能访问 `usize::MAX`，造成 UB
        
- 因此，索引必须显式转换：
    

```rust
v[i as usize]; // ✅ 明确转换
```

---

### 3. 示例

```rust
let n = 3;
let mut ans = vec![vec![0; n]; n];
let x: i32 = 1;
let y: i32 = 2;
let val = 42;

// 索引必须用 usize
ans[x as usize][y as usize] = val;
```

- `x as usize` 明确告诉编译器：“我确信 x >= 0，可以安全索引”
    
- 如果 x < 0，`as usize` 会 wrap-around，导致溢出（release 模式下不会 panic，但 debug 下可能 panic）
    

---

### 4. 小结

1. Rust 数组/Vec 索引 **只能是 usize**
    
2. 编译器不会自动把 `i32` / `i64` 转成 `usize`
    
3. 必须显式写 `as usize`，这是 Rust 的 **安全设计**
    
4. 如果有负数可能，最好先做边界检查再转换，否则会 wrap-around
    

---

如果你愿意，我可以画一个 **整数类型 → Vec 索引转换规则图**，清楚显示 i8/i32/i64 到 usize 的安全与 wrap-around 情况。

你希望我画吗？

## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
