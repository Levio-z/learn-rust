### 翻译

**动机（Motivation）：** 同时实现安全性与可控性  
- **安全性（Safety）：** 编译后的程序不会出错  
- **可控性（Control）：** 语言支持底层特性  
- **既有工作（Prior art）：** C/C++（不安全）

**本课程最佳适用内容（Best fit for this course）：**  
所有权（ownership）和生命周期（lifetime）完美体现了并发编程的核心思想

**阅读任务（Reading assignments）：**

- 阅读thebook。作业 1 与书中最终项目相关
- 阅读 _Rust by Example_
- 编程作业将使用 Rust
- 在提供的服务器上设置编程环境

### Rust 示例 1：迭代器失效（1/3）
```rust


fn main() {

let v = vec![1, 2, 3];

let p = &v[1];

v.push(4);

println!("v[1]: {}", *p);

}

```
 翻译

这个代码在 C++ 中可以编译，  
但在运行时可能会失败。 
- **`v.push(4)` 可能会导致 `v` 重新分配内存，**  从而使指针 `p` 失效（p指向一个未使用的内存变量）。底层向量缓冲区被重定位。

而这段代码在 Rust 中无法编译：  
- “无法将 `v` 可变借用，因为它同时也被不可变借用。”
### Rust 示例 1：迭代器失效（2/3）

```rust


fn main() {

let mut v = vec![1, 2, 3];

let p = &v[1];

v.push(4);

println!("v[1]: {}", *p);

}

```

**`v`：** 向量的所有者  
**`p`：** 对 `v` 进行不可变借用  
- 范围从 `let p = …` 到 `println!(...)`  
- **`v.push(4)`：** 对 `v` 进行可变借用，仅在这一行

编译失败是因为类型检查器检测到了向量的**共享可变访问（SMA，shared mutable accesses）**：  
即 `p` 与 `v.push` 同时访问。  
这正是程序可能出错的原因。

**问题：如何检测共 享可变访问（SMA）？**


### Rust 示例 1：迭代器失效（3/3）
```rust

fn main() {

let v = vec![1, 2, 3];

let p = &v[1];

v.push(4);

println!("v[1]: {}", *p);

}

```
计算每个所有者/借用者的“生命周期”

- **`v`**：L1–L5
    
- **`p`**：L3–L5
    
- **`v.push`**：L4
    

列出生命周期重叠的所有组合

- `v` 和 `p`（L3–L5）
    
- `v` 和 `v.push`（L4）
    
- `p` 和 `v.push`（L4）
    

移除借用者与被借用者的组合

- 剩下 `p` 和 `v.push`（L4）
    

移除不可变借用的组合

剩下的组合被视为共享可变访问（SMA）  
静态安全（Statically sound）：在**编译时**检测到所有 SMA

不完整：实际上并非所有组合都是共享可变访问（SMA）

### 用于分析共享可变访问（SMA）的所有权机制

**“所有权（Ownership）”**：一个主体访问和销毁资源的能力

- **独占（Exclusive）**：如果我拥有一个资源，别人不能拥有它
    
- **可借用（Borrowable）**：可以被单个主体可变借用，或被多个主体不可变借用
    
- **适用于并发（Fit for concurrency）**：每个线程都是一个主体
    

**执行纪律（Enforcing discipline）**：默认情况下，不允许对资源进行共享可变访问（SMA）

- **静态（Static）**：所有权纪律通过类型系统强制执行
    
- **易用（Easy to use）**：编译器会报告所有违反纪律的情况
    
- **正确（Correct）**：经过类型检查后，程序不会出错
    

**通过“内部可变性（interior mutability）”灵活处理纪律**

- **必要性（Necessary）**：在并发中，共享可变访问不可避免
    
- **模块化（Modular）**：通过安全的 API 封装实现
    
    - “仿佛（as if）不存在共享可变访问”

### Rust example 2: RefCell (1/4)
**背景（Context）：** 完全禁止共享可变访问（SMA）在某些场景下不现实，例如并发

**解决方案（Solution）：内部可变性（interior mutability）**  
通过安全的 API 封装 SMA，好像不存在 SMA 一样

**示例（Example）：`RefCell<T>`**  
在运行时（而非编译时）检查所有权

- `RefCell<T>::try_borrow()`：尝试不可变借用内部值
    
- `RefCell<T>::try_borrow_mut()`：尝试可变借用内部值
    

文档链接：

- [RefCell 文档](https://doc.rust-lang.org/stable/std/cell/struct.RefCell.html?utm_source=chatgpt.com)
    
- [Rust 官方书籍 — 内部可变性](https://doc.rust-lang.org/book/ch15-05-interior-mutability.html?utm_source=chatgpt.com)

```rust
**

fn f1() -> bool { true }

fn f2() -> bool { !f1() }

  

fn main() {

let mut v1 = 42;

let mut v2 = 666;

  

let p1 = if f1() { &v1 } else { &v2 };

  

if f2() {

let p2 = &mut v1;

*p2 = 37;

println!("p2: {}", *p2);

}

println!("p1: {}", *p1);

}

**
```

https://play.rust-lang.org/?version=stable&mode=debug&edition=2018&gist=c07efb0ed16980ef85d09568382114f9

假设 `f1()` 和 `f2()` 是复杂且互斥的条件（不是 `f1() && f2()`）

这是安全的，因为 `p1` 和 `p2` 没有发生别名（alias）

**编译错误原因**：  
类型检查器无法判断安全性  
由于条件过于复杂，出现错误信息：  
“无法将 `v1` 可变借用，因为它同时也被不可变借用。”

### **Rust example 2: RefCell (3/4)**

```rust
**

use std::cell::RefCell;

  

fn f1() -> bool { true }

fn f2() -> bool { !f1() }

  

fn main() {

let v1 = RefCell::new(42);

let v2 = RefCell::new(666);

  

let p1 = if f1() { &v1 } else { &v2 }

.try_borrow().unwrap();

  

if f2() {

let mut p2 = v1  
.try_borrow_mut().unwrap();

*p2 = 37;

println!("p2: {}", *p2);

}

  

println!("p1: {}", *p1);

}

**
```

所有权在运行时检查  
（`try_borrow()`、`try_borrow_mut()`）  
编译并按预期执行  
输出：“p1: 42”

如果 `f1() && f2()` 为真，`try_borrow_mut()` 会在**运行时**失败（而非编译时）  
报错信息：“thread 'main' panicked at 'called `Result::unwrap()` on an `Err` value'”

### **Rust example 2: RefCell (4/4)**

**内部可变性（Interior mutability）：** 在非 SMA 类型中封装 SMA

**安全 API：** 表面上没有 SMA

```rust
pub fn try_borrow_mut(&self) -> Result<RefMut<T>, BorrowMutError>
```

（不可变地借用 `self`）

**潜在不安全的实现：** 实际可能存在 SMA

```rust
... unsafe { &mut *self.value.get() }, …
```

[源码链接](https://doc.rust-lang.org/1.63.0/src/core/cell.rs.html?utm_source=chatgpt.com#1732)

**“Unsafe”**：桥接**无 SMA 的 API**和**含 SMA 的实现**

-   需要手动检查
    
-   应明确标注 `unsafe`

### **Rust example 3: Lock**

**[https://github.com/kaist-cp/cs431/blob/main/src/lock/api.rs#L105](https://github.com/kaist-cp/cs431/blob/main/src/lock/api.rs#L105)**

**![](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUdQ3Kf6zUxN5et_qv55kGdSSiZcw8MbbYz-eBEObcm0j5WZhpo0iPr2s54kC0P1-TPzSwSpoJB2CUtvBypDlEF5Prhoqw026HAFu9NRMfpRm6mvWvEerrPTweWGH4dHaV7nRXBGglzqADjwTVPYZKDNeR0EBoI=s2048?key=pFI0iQIu-AUwhliN9TiC1w)**


```
**// data: Lock<int>  
let data_guard = data.lock();  
let data_ref = data_guard.deref();  

drop(data_guard); // lock is released  
*data_ref = 666; // NOT COMPILED: deref target shall not outlive guard

```
### Rust 所有权类型总结（Summary of Rust’s ownership type）
**动机（Motivation）：** 实现对共享可变资源的安全性与可控性

**核心思想（Key ideas）：**

- **纪律（Discipline）**：默认禁止共享可变访问
    
- **内部可变性（Interior mutability）**：允许在受控方式下使用共享可变访问
    

**好处（Benefits）：**

- 静态分析共享可变访问的安全性  
    （适用于顺序程序和并发程序）
    
- 明确标注需要手动检查的代码
    

**我们的任务（What we’ll do）：**  
理解基于锁的并发编程，并结合安全 API 使用