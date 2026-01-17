---
tags:
  - permanent
---
## 1. 核心观点  

Rust 虽然没有类继承，但 **`Deref` 提供了类似“子类可当父类用”的语义**。  

- `str` 定义了字符串接口（只读视图抽象层）；
- `String` 扩展了该接口并添加堆管理能力；
- `Deref` 使得 **`String` 可以无缝被“视为” `str`。**


## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
当 `String` 实现了：
```rust
impl Deref<Target = str> for String
```
之后，编译器允许自动解引用：
```rust
fn print_str(s: &str) { println!("{s}"); }

let s = String::from("hello");
print_str(&s); // 自动 &String → &str
```
此处的“继承”可以理解为：
- `str` 定义了字符串接口（只读视图抽象层）；
- `String` 扩展了该接口并添加堆管理能力；
- `Deref` 使得 `String` 可以无缝被“视为” `str`。

这是一种 **组合 + 语义继承（via Deref）** 的多态方式。
## 4. 与其他卡片的关联  
- 前置卡片：
	- [Rust-自动解引用-基本概念-TOC](Rust-自动解引用-基本概念-TOC.md)
- 后续卡片：
- 相似主题：
	- [Rust-String和str的自动解引用](../../../../2.2%20类型系统、数据布局/2.2.1%20类型基础/Rust-String和str的自动解引用.md)
## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
- [x] 原子笔记整理
	- [x] 整理基本概念的笔记
	- [x] String中的自动解引用，String的本质
	- [x] 自动解引用规则

### 基本概念

####  **`Deref` 和 `DerefMut` trait**
`Deref` 定义：
```rust
trait Deref {
    type Target: ?Sized;
    fn deref(&self) -> &Self::Target;
}

```
`DerefMut` 定义：
```rust
trait DerefMut: Deref {
    fn deref_mut(&mut self) -> &mut Self::Target;
}

```

#### 自动解引用规则
在以下场景中，Rust **自动调用 `.deref()` 或 `.deref_mut()`**：
- 函数参数、方法接收器需要的类型与提供的引用类型不完全一致时。  
- 编译器能唯一确定目标类型
	- **返回值类型、赋值目标、函数签名**这些地方，先确定“需要的类型”。
- 一旦发现：
- 需要 `&U`，
- 你手上有 `&T` 且 `T: Deref<Target = U>`，
#### 自动借用调整
- 自动借用调整常与自动解引用配合工作
```rust
let b = Box::new(String::from("hi"));
let len = b.len();  
// 实际步骤：Box<String> --deref--> &String::len(
```
它就尝试自动解引用，直到类型匹配。
#### 底层原理
Rust 编译器中：

- 类型检查器（typeck）
    
- 自动解引用器（autoderef）
    

这两个组件共同工作：

> 在编译阶段识别出 `Deref` 和 `DerefMut` 实现，自动插入必要的 `.deref()` 调用。
#### 使用场景
这种自动转换适用场景主要有：  
`String` → `str`  
`Vec<T>` → `[T]`  
 `Box<T>` → `T`  
`Rc<T>`、`Arc<T>` → `T`
都是常见的 “智能指针 + 容器” 类型，它们通过 `Deref` 模拟出“像指针一样的行为”。
#### 具体案例
值类型&mut String， 返回类型&str
1️⃣ **`&mut String` → `String`** （通过 `*` 解引用）  
2️⃣ **`String` → `str`** （通过 `Deref`）  
3️⃣ **`str` → `&str`** （通过借用和 `Index`）