---
tags:
  - fleeting
---
## 1. 核心观点  
### Ⅰ. 概念层
使用 async fn do_something展开的代码如下，但是具体标注类型的async就不好展开了见详细部分，这样的展开不能send？
```
impl MyTrait for RespEnum {
    #[inline]
    async fn do_something(&self) {
        match self {
            RespEnum::RespNull(inner) => MyTrait::do_something(inner).await,
            RespEnum::RespSet(inner) => MyTrait::do_something(inner).await,
        }
    }
}
```
- [为什么不能和返回值是impl的使用](#为什么不能和返回值是impl的使用)
- [Rust-crate-strum-send传染性](Rust-crate-strum-send传染性.md)

### Ⅱ. 应用层




### Ⅲ. 实现层

### **IV**.原理层


## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### 为什么不能和返回值是impl的使用
```
enum E {
    A(A),
    B(B),
}

impl Trait for E {
    fn f(&self) {
        match self {
            E::A(x) => x.f(),
            E::B(x) => x.f(),
        }
    }
}
```


👉 **枚举分发 = 编译期生成 `match`，并把调用转发给具体实现**

```
trait Worker {
    fn work(&self) -> impl Future<Output = ()> + Send;
}
```

两个实现：
```
impl Worker for A {
    fn work(&self) -> impl Future<Output = ()> + Send {
        async { /* A */ }
    }
}

impl Worker for B {
    fn work(&self) -> impl Future<Output = ()> + Send {
        async { /* B */ }
    }
}

```
⚠️ **致命点**：

- `async {}` → **每一个 async block 都是不同的匿名 Future 类型**
    
- 即使：
    
    - `Output` 一样
        
    - 都是 `Send`
        
- **类型依然不同**


`enum_dispatch` 生成的代码，本质等价于：
```
impl Worker for E {
    fn work(&self) -> ??? {
        match self {
            E::A(x) => x.work(),
            E::B(x) => x.work(),
        }
    }
}

```
问题来了：
x.work() 的返回类型：
- 分支 A：FutureA
- 分支 B：FutureB

`match` 的所有分支，返回类型必须完全一致
### 为什么 async fn “看起来能用”

```
#[enum_dispatch]
trait Worker {
    async fn work(&self);
}
```

它**看起来能工作**，是因为：

- 你没有在 trait 层显式看到返回类型
- enum_dispatch 生成的 `async fn`：
    - 每个 impl 自己生成状态机
    - 外层 impl 也是一个 `async fn`
- **状态机被包了一层，类型不再暴露**
    
📌 但代价是：
- 你**无法声明 `Send`**
- 触发 `async_fn_in_trait` lint
- API 的并发语义是“隐式的、不稳定的”


| 方案                | 能否统一返回类型 | 说明              |
| ----------------- | -------- | --------------- |
| `impl Future`     | ❌        | 每个 impl 是不同匿名类型 |
| `async fn`        | ✅（被隐藏）   | 编译器生成外层状态机      |
| `Box<dyn Future>` | ✅        | 通过动态分发抹掉类型      |
| GAT + enum        | ❌        | 仍然是不同具体类型       |






## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
