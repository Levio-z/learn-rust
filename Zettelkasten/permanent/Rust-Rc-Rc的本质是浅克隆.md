---
tags:
  - permanent
---
## 1. 核心观点  

指向 **同一个堆上字符串数据**，拷贝开销极小（只修改计数）

## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### 本质
```rust
use std::rc::Rc;

let a = Rc::new(String::from("hello"));
let b = Rc::clone(&a); // ✅ 不复制数据，只增加引用计数
```

此时：

- `a` 和 `b` 都指向 **同一个堆上字符串数据**；
- `Rc` 的引用计数（`strong_count`）+1；
- 堆上数据只存在一份；
- 拷贝开销极小（只修改计数）。

### 源码(简化)
```rust
impl<T> Clone for Rc<T> {
    fn clone(&self) -> Rc<T> {
        // 安全地增加引用计数
        unsafe { self.inner().strong.set(self.inner().strong.get() + 1); }
        Rc { ptr: self.ptr }
    }
}
```

## 4. 与其他卡片的关联  
- 前置卡片：
	- [Rust-Rc-基本概念](Rust-Rc-基本概念.md)
- 后续卡片：
	- 
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
