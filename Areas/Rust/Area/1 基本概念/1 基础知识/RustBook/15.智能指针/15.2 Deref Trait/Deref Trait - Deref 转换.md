---
tags:
  - permanent
---


## 1. 核心观点  

- 当 `T: Deref<Target=U>`，可以将 `&T` 转换成 `&U`，也就是我们之前看到的例子
- 当 `T: DerefMut<Target=U>`，可以将 `&mut T` 转换成 `&mut U`
- 当 `T: Deref<Target=U>`，可以将 `&mut T` 转换成 `&U`

## 2. 背景/出处  

- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
```rust
struct MyBox<T> {
    v: T,
}

impl<T> MyBox<T> {
    fn new(x: T) -> MyBox<T> {
        MyBox { v: x }
    }
}

use std::ops::Deref;

impl<T> Deref for MyBox<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.v
    }
}

use std::ops::DerefMut;

impl<T> DerefMut for MyBox<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.v
    }
}

fn main() {
    let mut s = MyBox::new(String::from("hello, "));
    display(&mut s)
}

fn display(s: &mut String) {
    s.push_str("world");
    println!("{}", s);
}
```

- 要实现 `DerefMut` 必须要先实现 `Deref` 特征：`pub trait DerefMut: Deref`
- `T: DerefMut<Target=U>` 解读：将 `&mut T` 类型通过 `DerefMut` 特征的方法转换为 `&mut U` 类型，对应上例中，就是将 `&mut MyBox<String>` 转换为 `&mut String`

## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
	- [1.0.2 Rust-智能指针的Deref设计](../15.1Box/1.0.2%20Rust-智能指针的Deref设计.md)
	- [2.0 如何使用Deref](2.0%20如何使用Deref.md)
	- [Rust-自动解引用-Deref语义继承](../../../../2%20进阶/2.1%20所有权、生命周期和内存系统/2.1.3%20生命周期和引用/引用机制/Rust-自动解引用/Rust-自动解引用-Deref语义继承.md)
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
