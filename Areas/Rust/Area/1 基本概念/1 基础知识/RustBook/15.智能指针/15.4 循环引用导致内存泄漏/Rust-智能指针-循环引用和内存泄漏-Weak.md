---
tags:
  - note
---
## 1. 核心观点  
### Ⅰ. 概念层
背景：[创建循环引用是可能的](#创建循环引用是可能的)

`Weak` 通过 `use std::rc::Weak` 来引入，它具有以下特点:
- `Weak` **不持有所有权**，它仅仅保存一份指向数据的弱引用
	- [Weak原理](#Weak原理)
- 访问数据：需要通过 `Weak` 指针的 `upgrade` 方法实现，该方法返回一个类型为 `Option<Rc<T>>` 的值。如果资源已经被释放，则 `Option` 的值是 `None`。
- 由rc转换：可由 `Rc<T>` 调用 `downgrade` 方法转换成 `Weak<T>`
- 常用于解决循环引用的问题。

### Ⅱ. 应用层
适用场景：
通过下面案例这个对比，可以非常清晰的看出 `Weak` 为何这么弱，而这种弱恰恰非常适合我们实现以下的场景：
- 持有一个 `Rc` 对象的临时引用，并且不在乎引用的值是否依然存在
- 阻止 `Rc` 导致的循环引用，因为 `Rc` 的所有权机制，会导致多个 `Rc` 都无法计数归零
使用方式简单总结下：**对于父子引用关系，可以让父节点通过 `Rc` 来引用子节点，然后让子节点通过 `Weak` 来引用父节点**。

使用案例：
- [[# Weak案例1]]
- [Weak案例2](Rust-智能指针-循环引用和内存泄漏-Weak.md#Weak案例2)

注意事项
- [Weak的debug设计](#Weak的debug设计)只打印占位符，树 / 图结构可能无限展开
### Ⅲ. 实现层

### **IV**.原理层



## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### 创建循环引用是可能的

- Rust 的内存安全保证**使得意外创建永不清理的内存（称为_内存泄漏_）变得困难，但并非不可能**。
	- 完全防止内存泄漏并不是 Rust 的保证之一，这意味着内存泄漏在 Rust 中是内存安全的。
	- 我们可以看到 Rust 通过使用`Rc<T>`和`RefCell<T>`来允许内存泄漏：可以创建引用，其中项目在循环中相互引用。这会造成内存泄漏，因为循环中每个项目的引用计数永远不会达到 0，并且值永远不会被删除。
#### 创建循环引用的案例
```rust
use crate::List::{Cons, Nil};
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Debug)]
enum List {
    Cons(i32, RefCell<Rc<List>>),
    Nil,
}

impl List {
    fn tail(&self) -> Option<&RefCell<Rc<List>>> {
        match self {
            Cons(_, item) => Some(item),
            Nil => None,
        }
    }
}

fn main() {
let a = Rc::new(Cons(5, RefCell::new(Rc::new(Nil))));

    println!("a initial rc count = {}", Rc::strong_count(&a));
    println!("a next item = {:?}", a.tail());

    let b = Rc::new(Cons(10, RefCell::new(Rc::clone(&a))));

    println!("a rc count after b creation = {}", Rc::strong_count(&a));
    println!("b initial rc count = {}", Rc::strong_count(&b));
    println!("b next item = {:?}", b.tail());

    if let Some(link) = a.tail() {
        *link.borrow_mut() = Rc::clone(&b);
    }

    println!("b rc count after changing a = {}", Rc::strong_count(&b));
    println!("a rc count after changing a = {}", Rc::strong_count(&a));

    // Uncomment the next line to see that we have a cycle;
    // it will overflow the stack
    // println!("a next item = {:?}", a.tail());
}
```
这个类型定义看着复杂，使用起来更复杂！不过排除这些因素，我们可以清晰看出：

1. 在创建了 `a` 后，紧接着就使用 `a` 创建了 `b`，因此 `b` 引用了 `a`
2. 然后我们又利用 `Rc` 克隆了 `b`，然后通过 `RefCell` 的可变性，让 `a` 引用了 `b`

至此我们成功创建了循环引用`a`-> `b` -> `a` -> `b` ····

通过 `a.tail` 的调用，Rust 试图打印出 `a -> b -> a ···` 的所有内容，但是在不懈的努力后，`main` 线程终于不堪重负，发生了[栈溢出](https://course.rs/pitfalls/stack-overflow.html)。

以上的代码可能并不会造成什么大的问题，但是在一个更加复杂的程序中，类似的问题可能会造成你的程序不断地分配内存、泄漏内存，最终程序会不幸**OOM**，当然这其中的 CPU 损耗也不可小觑。

创建引用循环并不容易，但也不是不可能。如果您有包含`Rc<T>RefCell<T> <T>` 值或具有内部可变性和引用计数的类似嵌套类型组合，则必须确保不会创建循环；您不能依赖 Rust 来捕获它们。创建引用循环将是您程序中的一个逻辑错误，您应该使用自动化测试、代码审查和其他软件开发实践来尽量减少它。
### Weak案例1
```rust
use std::rc::Rc;
fn main() {
    // 创建Rc，持有一个值5
    let five = Rc::new(5);

    // 通过Rc，创建一个Weak指针
    let weak_five = Rc::downgrade(&five);

    // Weak引用的资源依然存在，取到值5
    let strong_five: Option<Rc<_>> = weak_five.upgrade();
    assert_eq!(*strong_five.unwrap(), 5);

    // 手动释放资源`five`
    drop(five);

    // Weak引用的资源已不存在，因此返回None
    let strong_five: Option<Rc<_>> = weak_five.upgrade();
    assert_eq!(strong_five, None);
}
```
### Weak案例2
换个角度思考这些关系，父节点应该拥有它的子节点：如果父节点被删除，它的子节点也应该被删除。然而，子节点不应该拥有它的父节点：如果我们删除一个子节点，它的父节点应该仍然存在。这就是弱引用的作用！
- 一个节点可以引用它的父节点，但并不拥有它的父节点。
```rust
use std::cell::RefCell;
use std::rc::{Rc, Weak};

#[derive(Debug)]
struct Node {
    value: i32,
    parent: RefCell<Weak<Node>>,
    children: RefCell<Vec<Rc<Node>>>,
}
fn main() {
    let leaf = Rc::new(Node {
        value: 3,
        parent: RefCell::new(Weak::new()),
        children: RefCell::new(vec![]),
    });

    println!("leaf parent = {:?}", leaf.parent.borrow().upgrade());

    let branch = Rc::new(Node {
        value: 5,
        parent: RefCell::new(Weak::new()),
        children: RefCell::new(vec![Rc::clone(&leaf)]),
    });

    *leaf.parent.borrow_mut() = Rc::downgrade(&branch);

    println!("leaf parent = {:?}", leaf.parent.borrow().upgrade());
}

```

```rust
use std::cell::RefCell;
use std::rc::{Rc, Weak};

#[derive(Debug)]
struct Node {
    value: i32,
    parent: RefCell<Weak<Node>>,
    children: RefCell<Vec<Rc<Node>>>,
}
fn main() {
    let leaf = Rc::new(Node {
        value: 3,
        parent: RefCell::new(Weak::new()),
        children: RefCell::new(vec![]),
    });

    println!(
        "leaf strong = {}, weak = {}",
        Rc::strong_count(&leaf),
        Rc::weak_count(&leaf),
    );

    {
        let branch = Rc::new(Node {
            value: 5,
            parent: RefCell::new(Weak::new()),
            children: RefCell::new(vec![Rc::clone(&leaf)]),
        });

        *leaf.parent.borrow_mut() = Rc::downgrade(&branch);

        println!(
            "branch strong = {}, weak = {}",
            Rc::strong_count(&branch),
            Rc::weak_count(&branch),
        );

        println!(
            "leaf strong = {}, weak = {}",
            Rc::strong_count(&leaf),
            Rc::weak_count(&leaf),
        );
    }

    println!("leaf parent = {:?}", leaf.parent.borrow().upgrade());
    println!(
        "leaf strong = {}, weak = {}",
        Rc::strong_count(&leaf),
        Rc::weak_count(&leaf),
    );
}
```

### Weak原理
#### 底层实现是weak_count
- `weak_count` ：不是将`Rc<T>`实例中的`strong_count`增加 1，而是调用 `Rc::downgrade`将`weak_count`增加1。Rc `Rc<T>`类型使用 `weak_count`用于跟踪存在多少个`Weak<T>`引用，类似于 `strong_count` 。
- 实现：**不增加引用计数，因此不会影响被引用值的释放回收**


#### [Weak 与 Rc 对比](https://course.rs/advance/circle-self-ref/circle-reference.html#weak-%E4%B8%8E-rc-%E5%AF%B9%E6%AF%94)

我们来将 `Weak` 与 `Rc` 进行以下简单对比：

| `Weak`                                | `Rc`                      |
| ------------------------------------- | ------------------------- |
| 不计数                                   | 引用计数                      |
| 不拥有所有权                                | 拥有值的所有权                   |
| 不阻止值被释放(drop)                         | 所有权计数归零，才能 drop           |
| 引用的值存在返回 `Some`，不存在返回 `None`          | 引用的值必定存在                  |
| 通过 `upgrade` 取到 `Option<Rc<T>>`，然后再取值 | 通过 `Deref` 自动解引用，取值无需任何操作 |
### Weak的debug设计
#### `Weak<T>` 的 Debug 设计原则

Rust 标准库对 `Weak<T>` 的 `Debug` 实现遵循一个非常重要的原则：

> **`Debug` 不应产生副作用，也不应隐式执行“语义性操作”**

而对 `Weak<T>` 来说：

- 想打印内部内容，**必须调用 `upgrade()`**
    
- `upgrade()`：
    
    - 是一次运行期有效性检查
        
    - 语义上可能失败（返回 `None`）
        
    - 甚至会影响你对生命周期的判断
        

因此：

- `Debug for Weak<T>` **不会自动调用 `upgrade()`**
    
- 只打印为一个占位标识：`(Weak)`

## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
