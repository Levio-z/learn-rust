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

[Rust example RefCell](../../../../Areas/Rust/Area/1%20基本概念/2%20进阶/2.8%20标准库/std/cell/RefCell/案例/Rust%20example%20RefCell.md)


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
C++是不安全的，因为它可以解引用，并且底层指针值可以泄漏

核心点在于：**`LockGuard` 内部持有锁的生命周期，并且其 `Deref` 返回的是 受借用检查器约束的安全引用，而不是裸指针本身。**

Rust的动机实在共享可变存在的情况下同时实现安全和控制。
### Rust 所有权类型总结（Summary of Rust’s ownership type）
**动机（Motivation）：** 实现对共享可变资源的安全性与可控性
**核心思想（Key ideas）：**
- **纪律（Discipline）**：默认禁止共享可变访问
- **内部可变性（Interior mutability）**：允许在受控方式下使用共享可变访问
**好处（Benefits）：**
- 静态分析共享可变访问的安全性  （适用于顺序程序和并发程序）
- 明确标注需要手动检查的代码
**我们的任务（What we’ll do）：**  
理解基于锁的并发编程，并结合安全 API 使用