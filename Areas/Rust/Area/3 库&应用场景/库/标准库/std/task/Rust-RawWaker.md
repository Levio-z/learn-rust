---
tags:
  - fleeting
---
## 1. 核心观点  
### Ⅰ. 概念层

RawWaker 允许任务执行器的实现者创建 Waker或者使用提供自定义唤醒行为的 LocalWaker 器。

它由数据指针和[虚函数指针表（vtable）](https://en.wikipedia.org/wiki/Virtual_method_table) 组成。 自定义 `RawWaker` 的行为。

使用 `RawWaker` 是不安全的。实现 [`Wake`](https://doc.rust-lang.org/alloc/task/trait.Wake.html) trait 是一种安全的替代方案，但需要分配内存。

`RawWaker` 是 Rust 异步运行时底层用于构造 `Waker` 的原始类型，其本质是一个 **手动实现运行时多态（runtime polymorphism）** 的结构。


它由两部分组成：

1. `data: *const ()` —— 一个类型擦除（type-erased）的数据指针
    
2. `vtable: &'static RawWakerVTable` —— 一个显式定义的“虚方法表”
	- 该表指定了当 RawWaker 被克隆、唤醒或被释放时应当调用的函数。

`RawWaker` 通过显式定义的 `RawWakerVTable`，手动实现了一套运行时多态机制。  
它利用 `*const ()` 进行类型擦除，使非泛型的 `RawWaker` 能承载任意任务状态；同时也将 **类型安全、生命周期管理、线程安全** 的全部责任交还给程序员。这种设计是为了满足异步运行时在零抽象成本、跨 crate、跨平台场景下的极致需求，但其危险性极高，因此只适合底层框架作者使用。
### Ⅱ. 应用层




### Ⅲ. 实现层

### **IV**.原理层


## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  

### RawWakerVTable 的布局与函数语义
`RawWakerVTable` 定义了四个函数指针：
```rust
pub struct RawWakerVTable {
    pub clone: unsafe fn(*const ()) -> RawWaker,
    pub wake: unsafe fn(*const ()),
    pub wake_by_ref: unsafe fn(*const ()),
    pub drop: unsafe fn(*const ()),
}
```
**语义解释**

- `clone`  
    用于 `Waker::clone`，必须正确处理引用计数或资源复制
    
- `wake`  
    唤醒任务，并 **消耗** waker（通常伴随 drop）
    
- `wake_by_ref`  
    唤醒任务，但 **不消耗** waker
    
- `drop`  
    释放底层资源（如减少引用计数、释放堆内存）
**本质**  
这与 C++ 的虚函数表在语义上完全等价，只是：

- 没有编译器自动生成
    
- 生命周期、别名、线程安全全部由程序员保证
### 为什么使用 `*const ()` 作为参数

核心原因：非泛型 + 任意类型支持
`RawWaker` 被设计为：

- **非泛型类型**
    
- 可被 `Waker`、`Future`、执行器统一使用
    
- 不暴露任何具体任务类型

如果使用 `&T` 或 `*const T`，则：

- `RawWaker` 必须是泛型
    
- 或必须引入 trait object（dyn Trait）

这两者都不符合设计目标。

解决方案：类型擦除

```rust
T ──> *const T ──> *const ()
```
- 在构造 `RawWaker` 时：  
    把真实数据的指针“抹掉类型”
    
- 在 vtable 函数中：  
    再 **按约定** 转回原始类型


### 四、data 指针的来源与生命周期约束

常见来源
```rust
Box<T>  ── Box::into_raw ──> *mut T ──> *const ()
Arc<T>  ── Arc::into_raw ──> *const T ──> *const ()
```

**为什么通常是堆分配**

- `Waker` 可被 clone、跨线程、跨 poll 调用
    
- 栈数据无法保证生命周期
    
- 堆分配是唯一合理选择

恢复方式（在 vtable 函数中）
```rust
let ptr = data as *const T;
let arc = Arc::from_raw(ptr);
```
必须严格匹配：
- `into_raw` ↔ `from_raw`
    
- clone / drop 中的引用计数语义一致

任何一次错误匹配都会导致：

- double free
    
- use-after-free
    
- 内存泄漏
### 五、RawWaker 的危险性本质

**危险点总结**

1. `unsafe fn`：编译器无法验证任何约束
    
2. `*const ()`：完全丢失类型信息
    
3. 手动管理引用计数
    
4. 必须保证多线程安全（Send + Sync 语义）
    
5. drop / wake / clone 的一致性必须 100% 正确
    

**为什么官方强烈不推荐手写**

因为它本质上是：

> “用 Rust 写了一套 C ABI 风格的虚函数系统”

一旦出错，**不是 panic，而是未定义行为（UB）**。

### 构造
为了提供该指针，我们将指针放入 [`RawWaker::new`](https://doc.rust-lang.org/stable/core/task/struct.RawWaker.html#method.new) （这个函数用于初始化 `RawWaker`）的 `data` 参数中。随后 `Waker` 会使用这个 `RawWaker` 的 `data` 调用 vtable 函数。

## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
