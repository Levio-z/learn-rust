---
tags:
  - note
---
## 1. 核心观点  
### Ⅰ. 概念层

**但是，在某些情况下，值在其方法中改变自身但对其他代码而言是不可变的，这很有用。
比如我们实现的特征是定义在外部库中，因此该签名根本不能修改。值此危急关头， 

RefCell

 闪亮登场：

### Ⅱ. 实现层

### Ⅲ. 原理层


## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
借用规则的结果是，当你有一个不可变的值时，你不能可变地借用它。例如，此代码不会编译：

```rust
fn main() {
    let x = 5;
    let y = &mut x;
}
```


    
- 让我们通过一个实际的例子来了解，我们可以使用`RefCell<T>`来改变一个不可变的值，并看看为什么它很有用。
    
    ```rust
    // 定义在外部库中的特征
    pub trait Messenger {
        fn send(&self, msg: String);
    }
    
    // --------------------------
    // 我们的代码中的数据结构和实现
    struct MsgQueue {
        msg_cache: Vec<String>,
    }
    
    impl Messenger for MsgQueue {
        fn send(&self, msg: String) {
            self.msg_cache.push(msg)
        }
    }
    
    ```
    
- 在报错的同时，编译器大聪明还善意地给出了提示：将 `&self` 修改为 `&mut self`，但是。。。我们实现的特征是定义在外部库中，因此该签名根本不能修改。值此危急关头， `RefCell` 闪亮登场：
    
    ```rust
    use std::cell::RefCell;
    pub trait Messenger {
        fn send(&self, msg: String);
    }
    
    pub struct MsgQueue {
        msg_cache: RefCell<Vec<String>>,
    }
    
    impl Messenger for MsgQueue {
        fn send(&self, msg: String) {
            self.msg_cache.borrow_mut().push(msg)
        }
    }
    
    fn main() {
        let mq = MsgQueue {
            msg_cache: RefCell::new(Vec::new()),
        };
        mq.send("hello, world".to_string());
    }
    ```
    
- 这个 MQ 功能很弱，但是并不妨碍我们演示内部可变性的核心用法：通过包裹一层 `RefCell`，成功的让 `&self` 中的 `msg_cache` 成为一个可变值，然后实现对其的修改。



## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
