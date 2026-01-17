---
tags:
  - permanent
title: Rust-自动借用（Auto-borrowing）
---

# 卡片笔记模板

## 1. 核心观点  
> 自动借用是**方法调用时的语法糖**，**自动**将将接收者**按照方法签名需求插入`&` 或 `&mut`**，**避免程序员显示书写引用**，让方法调用更加**ergonomics**（自然、简洁）。

## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### 3.1 历史演进
#### 3.1.1 早期自动借用的范围限制
- **硬编码实现**：编译器只为少量“常用内置集合类型”实现自动借用规则：
    - `Vec<T>`
    - `String`
    - `[T; N]` 数组
- **目标**：让常用集合类型的方法调用像其他高级语言一样简洁，不用写 `&v`。
- **总结：**
	- **支持**：`Vec<T>`、`String`、`[T; N]` 数组
	- **不支持**：用户自定义类型、智能指针（Box/Rc/Arc 等）
	- **表现**：调用方法时自动插入 `&`，只在方法调用的 **self 参数** 生效
	- **限制**：用途有限，编译器内部写死规则，用户类型扩展不了
#### 3.1.2 早期自动借用案例
**(1) Vec 自动借用**
```rust
fn main() {
    let v = vec![1, 2, 3];

    // 方法签名：fn len(&self) -> usize
    // v 的类型：Vec<i32>，并不是 &Vec<i32>
    // 编译器硬编码：自动插入 &v
    println!("{}", v.len()); // OK，早期 Rust 就能写

    // 用户自定义类型没有这个待遇
    struct MyVec(Vec<i32>);
    impl MyVec {
        fn len(&self) -> usize { self.0.len() }
    }

    let mv = MyVec(vec![1, 2, 3]);
    // mv.len(); // ❌ 早期不行，必须手写 &mv.len()
    println!("{}", (&mv).len()); // ✅
}
```
**(2) String 自动借用**
```rust
fn main() {
    let s = "hello".to_string();

    // fn len(&self) -> usize
    // 自动插入 &s
    println!("{}", s.len()); // OK

    // 如果是 Box<String>，不会触发
    let b = Box::new("hello".to_string());
    // b.len(); // ❌ 当时不行
    println!("{}", (&*b).len()); // ✅ 手动解引用 + 借用
}
```
**(3) 数组自动借用**
```rust
fn main() {
    let arr = [1, 2, 3];

    // fn len(&self) -> usize
    println!("{}", arr.len()); // OK，硬编码支持

    // 但是自定义 wrapper 不行
    struct MyArr([i32; 3]);
    impl MyArr {
        fn len(&self) -> usize { self.0.len() }
    }
    let ma = MyArr([1, 2, 3]);
    // ma.len(); // ❌ 当时不行
    println!("{}", (&ma).len()); // ✅
}
```
- **支持**：`Vec<T>`、`String`、`[T; N]` 数组
- **不支持**：用户自定义类型、智能指针（Box/Rc/Arc 等）
- **表现**：调用方法时自动插入 `&`，只在方法调用的 **self 参数** 生效

### 3.2 典型场景
```rust
let v = vec![1, 2, 3];
let len = v.len(); // 实际为 (&v).len() 自动借用
```
- 方法签名：`fn len(&self) -> usize`
- 接收者 `v` 类型：`Vec<i32>`（非引用）
- 自动借用：编译器在调用时插入 `&v` → 匹配 `&self` 参数
### 3.3 注意

**自动借用（autoref / auto-borrowing）只在方法调用的接收者（第一个参数，也就是 `self`）上起作用**，不会自动应用到普通函数的其他参数。
## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：[Rust-类型转换硬编码](Rust-类型转换硬编码.md)

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
