---
tags:
  - permanent
---
## 1. 核心观点  

在 Rust 中，方法调用的解析逻辑依赖于调用对象的类型和方法的来源，核心可以分为 **静态分派** 和 **动态分派** 两类。

## 2. 背景/出处  
- 来源：
	- [方法调用表达式](https://doc.rust-lang.org/reference/expressions/method-call-expr.html)
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  

### Rust 方法调用解析与分派机制

在 Rust 中，方法调用的解析逻辑依赖于调用对象的类型和方法的来源，核心可以分为 **静态分派** 和 **动态分派** 两类。

---

### 1. 静态分派（Static Dispatch）

#### 原理
- 当 **调用对象的具体类型在编译期已知** 时，编译器可以直接确定调用的方法。
- 编译器会：
    1. 查找该类型本身定义的关联方法（`impl T { fn foo(...) }`）。
	    1. 这部分不依赖trait
    2. 查找该类型实现的 trait 的方法（`impl Trait for T`）。
	    1.  只有作用域可见的 trait，且类型实现了该 trait，才能解析 trait 方法。
    3. 根据需要进行泛型单态化（monomorphization），生成针对具体类型的机器码。
- **性能高**：调用开销为普通函数调用。
- **泛型支持**：在泛型函数中，只要类型在实例化时已知，也会静态分派。
- **编译期检查**：类型安全、方法存在性在编译期完全确定。
#### 示例

```rust
trait TraitA {
    fn hello(&self);
}

struct S;
impl TraitA for S {
    fn hello(&self) { println!("Hello from S"); }
}

fn main() {
    let s = S;
    s.hello(); // 静态分派，编译器直接确定 S::hello
    call_hello(s);//
}

fn call_hello<T: TraitA>(x: T) {
    x.hello();
}
```
- 编译器知道 `T = S`。
- 会生成一段机器码，直接调用 `S::hello`。
- 如果再调用 `call_hello(AnotherStruct)`，编译器会生成 **另一份机器码**。

> 这就是 Rust 泛型的 **零成本抽象**：在运行时没有泛型开销，但编译期会多生成针对不同类型的函数代码。
---

### 2. 动态分派（Dynamic Dispatch）

#### 原理

- 当方法调用通过 **特征对象**（trait object）进行时，例如 `&dyn Trait` 或 `Box<dyn Trait>`：
    1. 编译器无法在编译期确定具体类型。
    2. 方法调用通过 **虚表（vtable）查找**，在运行时选择正确的实现。
- 调用对象是间接引用（trait object）时：
    - `let obj: &dyn Trait = &S;`
    - `obj.hello();` → 动态分派，依赖运行时 vtable。

- **用 `&dyn Trait` 时，本质上就是 trait object，编译器永远无法在编译期确定其具体类型**，因此这种情况下调用总是 **动态分派**。
- **即使我们知道在 `main` 里指向的是 S，也不影响动态分派**。编译器只看静态类型 `&dyn TraitA`。
#### 特点
- **灵活性高**：可在运行时决定实际调用的方法。
- **开销略高**：每次调用都需通过虚表查找。
- **支持多态**：允许不同类型的对象通过同一 trait 调用方法。

#### 示例

```rust
trait TraitA {
    fn hello(&self);
}

struct S;
impl TraitA for S {
    fn hello(&self) { println!("Hello from S"); }
}

fn greet(obj: &dyn TraitA) {
    obj.hello(); // 动态分派，通过 vtable 调用 S::hello
}

fn main() {
    let s = S;
    greet(&s);
}
```

---

### 3. 方法查找顺序

Rust 在调用方法时的查找顺序：

1. **检查自类型的方法**：
    - `impl T` 中定义的关联方法。
2. **检查 trait 方法**：
    - 已导入作用域的 trait。
	    - 如果类型实现了 trait，trait 中的默认方法或覆盖方法可调用。
		    - Trait 中可以提供 **默认实现**，类型实现 trait 时 **可以不覆盖**。
				- 调用对象如果没有自己的实现，就会使用 trait 提供的默认方法。
			- 类型可以提供 **自己的实现**，覆盖 trait 的默认实现。
				- 调用对象会调用类型自己的方法，而不是默认方法。
3. **自动引用/解引用（autoderef/autoref）**：
    - 编译器会自动尝试对左值进行 `&`、`&mut`、`*` 转换以匹配方法签名。
4. **泛型约束匹配**：
    - 如果是泛型类型 `T`，需要满足 `T: Trait` 的 trait bound 才能调用 trait 方法。

> ⚠️ 如果 trait bound 不满足，就会报错：

```text
error[E0277]: the trait bound `T: TraitB` is not satisfied
```

## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  

