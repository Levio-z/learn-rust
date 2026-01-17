---
tags:
  - permanent
---
## 1. 核心观点  

静态提升发生在**编译期**，它的目的是将一个本来应该在栈上创建的**临时值**，转存到**只读静态数据区**，从而允许对其的**共享引用**获得 ’static 生命周期。

字符串字面量是**唯一**一个**不需要**显式使用 `&` 运算符**，其表达式结果就已经是 $\&'static \text{str}$ 类型**的字面量。

静态提升的地址应该保持不变！
[代码验证地址](https://github.com/learn-rust-projects/rust-lab/blob/master/std/fs/file/src/static_promotion.rs)
## 2. 展开说明  

## 静态提升（Static Promotion）的所有主要场景

### 场景 1: 对字面量（Literal）的共享引用

这是最常见的情况，您已经非常熟悉：

- **核心：** 对基本类型、数组、元组的字面量直接取共享引用 `&`。
    
- **示例：**
    
    Rust
    
    ```
    let a: &'static i32 = &42;                 // 基本数字字面量
    let b: &'static [i32] = &[1, 2, 3];        // 数组字面量
    let c: &'static (&str, i32) = &("key", 10); // 元组字面量
    ```
    

### 场景 2: 对常量表达式（const Expressions）的共享引用

如果表达式的结果可以在**编译时完全确定**（即是 const 表达式），并且对其取共享引用 `&`，也会触发静态提升。

- **核心：** 编译器执行 **const 求值**，然后将结果存储为静态数据。
    
- **示例：**
    
    Rust
    
    ```
    let x: &'static u32 = &(5 + 7);               // 编译时计算出 12
    let y: &'static [i32] = &([1, 2].concat());   // 编译时执行 concat (假设 const fn)
    
    // 对 const 变量取引用也会触发（因为 const 相当于行内替换）
    const VAL: u32 = 99;
    let z: &'static u32 = &VAL; 
    ```
    

### 场景 3: 对空结构体或空枚举的引用（Zero-Sized Types, ZSTs）

空结构体或空枚举（它们在内存中不占用空间）的字面量引用也可以是 ’static。

- **核心：** 零大小类型本质上没有“数据”需要存储，它们的引用可以被视为 ’static 的。
    
- **示例：**
    
    Rust
    
    ```
    struct Empty;
    let e: &'static Empty = &Empty;
    
    enum Void {}
    // 注意：Option<T> 的 None 变体也是 ZST
    let n: &'static Option<i32> = &None;
    ```
    

### 场景 4: 结构体字面量的引用

对由常量数据构成的结构体字面量取共享引用，也会被提升。

- **核心：** 结构体字段也必须是字面量或 const 表达式的结果。
    
- **示例：**
    
    Rust
    
    ```
    struct Point { x: i32, y: i32 }
    let p: &'static Point = &Point { x: 10, y: 20 };
    ```
    

---

## 静态提升的限制条件（不会提升的场景）

静态提升的目的是生成一个**不可变且安全**的 ’static 引用。因此，任何可能导致**可变性**或**运行时错误**的表达式都**不会**被提升。

### 限制 1: 包含内部可变性（Internal Mutability）的类型

如果字面量中包含以下类型，则不会被提升，因为 ’static 共享引用必须保证不可变性：

- **`UnsafeCell`** 及其包装类型（如 `Cell<T>`, `RefCell<T>`）。
    
- **示例：**
    
    Rust
    
    ```
    use std::cell::Cell;
    // let c: &'static Cell<i32> = &Cell::new(1); // 编译错误！不能提升
    ```
    
    _（注意：虽然 `Cell::new(1)` 是一个 const fn，但它的结果包含了 `UnsafeCell`，被限制了。）_
    

### 限制 2: 包含需要运行时分配或计算的类型

以下类型需要运行时环境，因此不能被提升到编译期的静态区：

- **堆分配的类型：** `String::new()`, `Vec::new()`, `Box<T>`。
    
- **示例：**
    
    Rust
    
    ```
    // let s: &'static String = &String::from("a"); // 错误，需要堆分配
    // let v: &'static Vec<i32> = &vec![1, 2];      // 错误，需要堆分配
    ```
    

### 限制 3: 可变引用

`&mut T` 可变引用**永远不会**被提升为 &′staticmut T。

- **原因：** ’static 可变引用要求在整个程序运行期间都可以被修改，这在并发环境下是极其危险的，并且与编译器的安全哲学相悖。
    
- **示例：**
    
    Rust
    
    ```
    // let m: &'static mut i32 = &mut 5; // 编译错误！
    ```

## 3. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 4. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
