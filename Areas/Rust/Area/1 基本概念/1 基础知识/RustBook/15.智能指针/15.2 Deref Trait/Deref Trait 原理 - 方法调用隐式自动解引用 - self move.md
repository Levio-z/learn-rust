---
tags:
  - permanent
---


## 1. 核心观点  

**只有“唯一拥有者”才可能 move。**



## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### 1️⃣ `Box<T>`：可以

`let x = Box::new(String::from("hi")); 
x.into_bytes(); // ✅`
原因：
- auto-deref：`Box<String>` → `String`
- `Box` 拥有 `String`
- 可以 move 出来

等价于：

`String::into_bytes(*x)`

---

### 2️⃣ `Rc<T>`：不可以

`let x = Rc::new(String::from("hi")); x.into_bytes(); // ❌`

原因：

- auto-deref：`Rc<String>` → `String`
- **但 Rc 不允许 move 出内部值**
- `self` 的 receiver 无法构造
    

不是 auto-deref 失败，是 **receiver 不可满足**

---

### 3️⃣ `&T`：不可以（更直观）

`let s = String::from("hi"); let r = &s; r.into_bytes(); // ❌`

原因：

- 你手里只有 `&String`
    
- `self` 需要 `String`
    
- auto-deref 不能“复制一个 String 给你”

### “判断表”（非常实用）

|类型|能否 move 出 `T`|为什么|
|---|---|---|
|`Box<T>`|✅|唯一所有者|
|`Option<Box<T>>`|✅|可被 take|
|`Rc<T>`|❌|共享所有权|
|`Arc<T>`|❌|共享 + 多线程|
|`&T`|❌|借用，不拥有|
|`&mut T`|❌|借用，不拥有|
|`RefCell<T>`|❌|内部可变，不是所有权容器|
|`Pin<Box<T>>`|❌（一般）|Pin 语义禁止 move|
|`ManuallyDrop<T>`|✅（但危险）|手动管理 drop|

你会发现一个规律：  
**只有“唯一拥有者”才可能 move。**
### 案例
```rust
use std::rc::Rc;

fn main() {
    let x = Box::new(Box::new(Box::new(String::from("hi")))); 
    let y = x.into_bytes(); // ✅

    println!("inner_rc_clone: {:?}", y);
}

```

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
 
  
