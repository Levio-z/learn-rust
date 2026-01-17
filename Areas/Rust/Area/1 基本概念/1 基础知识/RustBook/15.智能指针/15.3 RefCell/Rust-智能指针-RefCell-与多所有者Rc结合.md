---
tags:
  - note
---
## 1. 核心观点  
### Ⅰ. 概念层

`RefCell<T>` 常用用法是与 `Rc<T>` 结合使用。请记住， `Rc<T>` 允许您为某些数据设置多个所有者，但它仅提供对该数据的不可变访问权限。如果您有一个包含 `RefCell<T>` `Rc<T>` 3CT>，您就可以获得一个可以拥有多个所有者_并且_您可以修改的值！

### Ⅱ. 应用层
- **应用场景：跟踪在只允许使用不可变值的上下文中使用它**。
    - 在一个不可变借用，内部可变借用的修改一个字段，外部仍是不可变的借用
- 使用注意事项：
	- 不适合多线程
	- 借用规则运行期强制执行
	- **开发后期发现错误以及运行时性能略微下降**
- 案例：[Rust-智能指针-内部可变性-案例](Rust-智能指针-内部可变性-案例.md)

### Ⅲ. 实现层


### **IV**.原理层
- [Rust-设计模式-内部可变性-基本概念](../../../../3%20设计模式/Rust-设计模式-内部可变性-基本概念.md)
- [Rust-运行期进行可变借用检查](Rust-运行期进行可变借用检查.md)


## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
```rust
#[derive(Debug)]
enum List {
    Cons(Rc<RefCell<i32>>, Rc<List>),
    Nil,
}

use crate::List::{Cons, Nil};
use std::cell::RefCell;
use std::rc::Rc;

fn main() {
    let value = Rc::new(RefCell::new(5));

    let a = Rc::new(Cons(Rc::clone(&value), Rc::new(Nil)));

    let b = Cons(Rc::new(RefCell::new(3)), Rc::clone(&a));
    let c = Cons(Rc::new(RefCell::new(4)), Rc::clone(&a));

    *value.borrow_mut() += 10;

    println!("a after = {a:?}");
    println!("b after = {b:?}");
    println!("c after = {c:?}");
}

```
### [性能损耗](https://course.rs/advance/smart-pointer/cell-refcell.html#%E6%80%A7%E8%83%BD%E6%8D%9F%E8%80%97)

相信这两者组合在一起使用时，很多人会好奇到底性能如何，下面我们来简单分析下。

首先给出一个大概的结论，这两者结合在一起使用的性能其实非常高，大致相当于没有线程安全版本的 C++ `std::shared_ptr` 指针，事实上，C++ 这个指针的主要开销也在于原子性这个并发原语上，毕竟线程安全在哪个语言中开销都不小。

### [内存损耗](https://course.rs/advance/smart-pointer/cell-refcell.html#%E5%86%85%E5%AD%98%E6%8D%9F%E8%80%97)

两者结合的数据结构与下面类似：

```rust

struct Wrapper<T> {
    // Rc
    strong_count: usize,
    weak_count: usize,

    // Refcell
    borrow_count: isize,

    // 包裹的数据
    item: T,
}

```

从上面可以看出，从对内存的影响来看，仅仅多分配了三个`usize/isize`，并没有其它额外的负担。
### [CPU 损耗](https://course.rs/advance/smart-pointer/cell-refcell.html#cpu-%E6%8D%9F%E8%80%97)

从 CPU 来看，损耗如下：

- 对 `Rc<T>` 解引用是免费的（编译期），但是  带来的间接取值并不免费
- 克隆 `Rc<T>` 需要将当前的引用计数跟 `0` 和 `usize::Max` 进行一次比较，然后将计数值加 1
- 释放（drop） `Rc<T>` 需要将计数值减 1， 然后跟 `0` 进行一次比较
- 对 `RefCell` 进行不可变借用，需要将 `isize` 类型的借用计数加 1，然后跟 `0` 进行比较
- 对 `RefCell` 的不可变借用进行释放，需要将 `isize` 减 1
- 对 `RefCell` 的可变借用大致流程跟上面差不多，但是需要先跟 `0` 比较，然后再减 1
- 对 `RefCell` 的可变借用进行释放，需要将 `isize` 加 1

其实这些细节不必过于关注，只要知道 CPU 消耗也非常低，甚至编译器还会对此进行进一步优化！

### [CPU 缓存 Miss](https://course.rs/advance/smart-pointer/cell-refcell.html#cpu-%E7%BC%93%E5%AD%98-miss)

唯一需要担心的可能就是这种组合数据结构对于 CPU 缓存是否亲和，这个我们无法证明，只能提出来存在这个可能性，最终的性能影响还需要在实际场景中进行测试。

总之，分析这两者组合的性能还挺复杂的，大概总结下：

- 从表面来看，它们带来的内存和 CPU 损耗都不大
- 但是由于 `Rc` 额外的引入了一次间接取值（），在少数场景下可能会造成性能上的显著损失
- CPU 缓存可能也不够亲和
### 多线程

与 `Rc<T>` 类似， `RefCell<T>` 仅适用于单线程场景，如果在多线程环境中使用，将会产生编译时错误。我们将在第 16 章讨论如何在多线程程序中使用 `RefCell<T>` 的功能。

## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
