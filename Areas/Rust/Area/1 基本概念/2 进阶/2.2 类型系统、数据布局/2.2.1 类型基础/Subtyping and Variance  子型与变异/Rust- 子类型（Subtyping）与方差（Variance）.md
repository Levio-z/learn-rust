---
tags:
  - draft
---
## 1. 核心观点  

Rust 通过 `Deref` 和 `DerefMut` trait 的 **自动解引用（deref coercion）机制**，允许某些类型之间的**隐式借用转换**。

## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  

### Rust 子类型（Subtyping）与方差（Variance）原子笔记（高密度版）

---

### 1. 子类型基本概念

- **子类型是隐式发生的**，可出现在类型检查或类型推断的任意阶段。
- Rust 中子类型 **仅存在两种来源**：
    1. **基于生命周期的子类型**（`'static: 'a` → `&'static T` 是 `&'a T` 的子类型）
    2. **基于高阶生命周期（HRTB）的子类型**（`for<'a> fn(..)`）
除生命周期外，Rust 类型系统是 _不支持一般子类型_ 的（例如 `Dog` 不是 `Animal`）。

---

### 2. 生命周期子类型：`'static` → `'a`

示例：

```rust
fn bar<'a>() {
    let s: &'static str = "hi";
    let t: &'a str = s; // 允许
}
```

原因：若 `'static: 'a`（`'static` 活得更久），则：

```
&'static T  <:  &'a T
```

---

### 3. 高阶生命周期子类型（HRTB）

高阶生命周期（`for<'a> ...`）在函数指针与 trait 对象中形成子类型规则。

示例 1：从高阶到具体生命周期的替换：

```rust
let s: &(for<'a> fn(&'a i32) -> &'a i32) = &(|x| x);
let t: &(fn(&'static i32) -> &'static i32) = s; // allowed
```

示例 2：trait 对象同理：

```rust
let s: &(dyn for<'a> Fn(&'a i32) -> &'a i32) = &|x| x;
let t: &(dyn Fn(&'static i32) -> &'static i32) = s; // allowed
```

示例 3：高阶生命周期之间：

```rust
let s: &(for<'a,'b> fn(&'a i32,&'b i32)) = &(|_,_|{});
let t: &(for<'c> fn(&'c i32,&'c i32)) = s; // allowed
```

---

### 4. 方差（Variance）的含义

泛型类型 `F<T>` 的**方差**描述了参数 `T` 的子类型关系如何“传递”到整个类型 `F<T>`：

- **协变（covariant）**：  
    `T <: U ⇒ F<T> <: F<U>`
    
- **逆变（contravariant）**：  
    `T <: U ⇒ F<U> <: F<T>`
    
- **不变（invariant）**：  
    无法推导子类型关系
    

---

### 5. 内置类型的方差表

|类型|`'a` 方差|`T` 方差|
|---|---|---|
|`&'a T`|协变|协变|
|`&'a mut T`|协变|不变|
|`*const T`|—|协变|
|`*mut T`|—|不变|
|`[T]`、`[T; n]`|—|协变|
|`fn() -> T`|—|协变|
|`fn(T) -> ()`|—|逆变|
|`UnsafeCell<T>`|—|不变|
|`PhantomData<T>`|—|协变|
|`dyn Trait<T> + 'a`|协变|不变|

---

### 6. 自定义类型的方差（由字段决定）

规则：  
**字段类型中 T 的方差 → 决定 F 的方差；若 T 在不同方差位置出现 → 不变。**

示例结构体：

```rust
struct Variance<'a,'b,'c,T,U:'a> {
    x: &'a U,               // 'a 协变，U 协变
    y: *const T,            // T 协变
    z: UnsafeCell<&'b f64>, // 'b 不变
    w: *mut U,              // U 不变 ⇒ 整个结构体在 U 上不变
    f: fn(&'c())-> &'c(),   // 'c 既逆变又协变 ⇒ 'c 不变
}
```

结果：

- `'a`：协变
    
- `T`：协变
    
- `'b`：不变
    
- `'c`：不变
    
- `U`：不变
    

---

### 7. 复合类型（非结构体）中的方差位置独立计算

位置独立 ⇒ 不会“合并”方差。

示例 1：

```rust
fn generic_tuple<'short,'long:'short>(x: (&'long u32, UnsafeCell<&'long u32>)) {
    let _: (&'short u32, UnsafeCell<&'long u32>) = x;
}
```

- 元组第 0 位是协变位置 ⇒ `'long` 可缩短为 `'short`
    
- 第二位在 `UnsafeCell` 中 ⇒ 不变位置 ⇒ 不能缩短
    

示例 2：

```rust
fn takes_fn_ptr<'short,'middle:'short>(f: fn(&'middle())-> &'middle()) {
    let _: fn(&'static())-> &'short() = f;
}
```

- 函数入参位置：逆变 ⇒ `&'middle()` 可扩展为 `'static`
    
- 函数返回值位置：协变 ⇒ `&'middle()` 可缩短为 `'short`
    

---

### 总结

Rust 子类型体系极度受限，仅存在：

1. 生命周期子类型（`'static <: 'a`）
    
2. 高阶生命周期子类型（HRTB）
    

泛型类型的方差由 **其字段中参数的使用位置** 决定：

- 协变：安全地“缩短生命周期”
    
- 逆变：安全地“延长生命周期”
    
- 不变：不能发生子类型
    

掌握这些规则是理解以下核心特性的重要基础：

- 引用规则（`&T` vs `&mut T`）
    
- trait 对象与 HRTB (`for<'a>`)
    
- unsafe 代码安全边界
    
- PhantomData 的行为
    
- 自定义智能指针与容器设计
    

---

### 学习方法论

1. **先画生命周期图**：标出 `'long`、`'short` 之间关系，再推导子类型。
    
2. **找方差位置**：判断字段中是否是协变/逆变/不变。
    
3. **手写推导式**：  
    例如：
    
    ```
    'static: 'a ⇒ &'static T <: &'a T
    ```
    
4. **构造反例**：修改字段，观察编译器如何拒绝非法子类型。
    

---

### 练习题

1. 判断以下类型在 T 上是否协变 / 逆变 / 不变：
    
    - `Box<T>`
        
    - `RefCell<T>`
        
    - `Arc<fn(T)->T>`
        
    - `struct S<'a, T> { x: fn(&'a T) }`
        
2. 推导以下代码是否允许赋值，并说明原因：
    

```rust
fn test<'short,'long:'short>(x: &'long i32) {
    let _: &'short i32 = x;
}
```

3. 分析方差：
    

```rust
struct X<'a, T> {
    a: &'a mut T,
    b: *const T,
    c: PhantomData<&'a T>,
}
```

分别说明 `'a` 和 `T` 的方差。

---

如需，我可以继续为你构建 **更细颗粒度的子类型/方差原子笔记库** 或配套练习解析。

## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
- [x] 原子笔记整理
	- [x] 整理基本概念的笔记
	- [x] String中的自动解引用，String的本质
	- [x] 自动解引用规则

