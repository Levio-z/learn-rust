---
tags:
  - permanent
---
## 1. 核心观点  

### 定义与作用
```
let pi: Result<f32, _> = "3.14".parse();
let log_pi = pi.unwrap_or(1.0).log(2.72);
```
_方法调用_由一个表达式（ _接收器_ ）后跟一个点、一个表达式路径段和一个括号中的表达式列表组成。

## 2. 背景/出处  
- 来源：
	- [方法调用表达式](https://doc.rust-lang.org/reference/expressions/method-call-expr.html)
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### 方法解析

方法调用被解析为特定特征上的关联[方法](https://doc.rust-lang.org/reference/items/associated-items.html#methods) ，如果左侧的确切`自`类型已知，则静态分派到方法，如果左侧表达式是间接[特征对象](https://doc.rust-lang.org/reference/types/trait-object.html) ，则动态分派。

### 候选接收器类型的构建（逐步）

1. 从接收者表达式的静态类型开始，把该类型记为第一个候选 `T0`。
2. 重复对当前类型执行 **解引用（`*`）**，把每次得到的类型 `Ti` 依次加入候选列表，直到不能再解引用为止。
3. 对每个被加入的候选类型 `T`，**紧接着**把 `&T` 和 `&mut T` 也加入列表（即在 `T` 之后分别加入不可变借用和可变借用候选）。
4. 如果最后得到的候选类型能进行 **unsized coercion**（例如固定长度数组 `[T; N]` → 切片 `[T]`），把强制后的类型以及它的 `&`/`&mut` 形式也加入。
5. 最终得到一个有序的候选接收器类型序列，编译器对这个序列按顺序搜索匹配的方法（更具体的类型先搜索）。

**示例（`Box<[i32;2]>`）**：  
候选顺序（简化）：

- `Box<[i32;2]>`， `&Box<[i32;2]>`， `&mut Box<[i32;2]>`
- `[i32;2]`（解引用）， `&[i32;2]`， `&mut [i32;2]`
- `[i32]`（unsized coercion）， `&[i32]`， `&mut [i32]`

```
String
&String
&mut String
↓  String 不提供该方法，就继续 deref
str
&str
&mut str
```
### 在每个候选类型上查找方法（优先顺序与范围）

对某个候选类型 `T`，查找可见方法的顺序与来源通常为：

1. **T 的固有方法（inherent impl）**：直接在 `impl T { ... }` 中定义的方法。固有方法优先级最高（对该具体类型明确实现的函数）。
2. **T 实现的可见 trait 的方法**：
	-  **强制约束（bound）方法**：必须优先使用，保证泛型代码在所有符合约束的类型上都可用。
	- **其它可见 trait 方法**：仅在 bound 外可见时才可调用，不是必需的。
3. **当且仅当上面没有找到合适方法时**，才会考虑进一步的候选（下一个 `T` 或其 `&T`/`&mut T` 版本）。
4. **若多个 trait 都提供同名方法且都可见**，会发生二义性——编译器会报错并要求用完全限定语法（`<Type as Trait>::method(&self, ...)`）或者通过类型标注消除歧义。
5. **动态分配或者静态分配的bound约束**：若接收者类型是 trait 对象（例如 `&dyn Trait` 或 `Box<dyn Trait>`），编译器在编译时找到该 trait 的对象安全方法并在运行时通过 vtable 调用（动态分派）。**若方法来自实现该 trait 的其它 trait 或固有 impl，则不适用于 trait 对象（除非有相应的对象安全 trait）**。


### 类型歧义与完全限定语法

当多个候选（固有 impl 或多个 trait）都提供同名方法并且都可见时，编译器无法决定时会报错。解决方法：

- 使用完全限定语法：`<Type as Trait>::method(receiver, args...)`。
- 改变作用域（隐藏/导入不同的 trait）。
	- **作用域决定哪些 trait 方法可见**，不可见的 trait 方法不会被考虑。
- 改变接收者的静态类型（类型注解）使候选集合不同。
```
    // 改变类型注解，接收者是 trait 对象
    let obj: &dyn TraitA = &s;
```





## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
	- [Rust-方法调用-调用解析与分派机制](Rust-方法调用-调用解析与分派机制.md)
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
- [ ] 原子笔记整理
	- [ ] 整理基本概念的笔记
	- [ ] String中的自动解引用，String的本质
	- [ ] 自动解引用规则
		- [ ] x
		- [ ] x
		- [ ] x
- [ ] 验证这个观点的边界条件  

### 基本概念

####  **`Deref` 和 `DerefMut` trait**
`Deref` 定义：
```rust
trait Deref {
    type Target: ?Sized;
    fn deref(&self) -> &Self::Target;
}

```
`DerefMut` 定义：
```rust
trait DerefMut: Deref {
    fn deref_mut(&mut self) -> &mut Self::Target;
}

```
[Rust-String和str的自动解引用](../../../2.2%20类型系统、数据布局/2.2.1%20类型基础/Rust-String和str的自动解引用.md)
#### 自动借用调整
- 自动借用调整常与自动解引用配合工作
```rust
let b = Box::new(String::from("hi"));
let len = b.len();  
// 实际步骤：Box<String> --deref--> &String::len(
```
它就尝试自动解引用，直到类型匹配。
#### 底层原理
Rust 编译器中：

- 类型检查器（typeck）
    
- 自动解引用器（autoderef）
    

这两个组件共同工作：

> 在编译阶段识别出 `Deref` 和 `DerefMut` 实现，自动插入必要的 `.deref()` 调用。
#### 使用场景
这种自动转换适用场景主要有：  
`String` → `str`  
`Vec<T>` → `[T]`  
 `Box<T>` → `T`  
`Rc<T>`、`Arc<T>` → `T`
都是常见的 “智能指针 + 容器” 类型，它们通过 `Deref` 模拟出“像指针一样的行为”。
#### 具体案例
值类型&mut String， 返回类型&str
1️⃣ **`&mut String` → `String`** （通过 `*` 解引用）  
2️⃣ **`String` → `str`** （通过 `Deref`）  
3️⃣ **`str` → `&str`** （通过借用和 `Index`）