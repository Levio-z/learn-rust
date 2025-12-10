---
tags:
  - permanent
---

## 1. 核心观点  

_关联的常量_是与类型关联[的常量](https://doc.rust-lang.org/reference/items/constant-items.html) 。

1. 关联常量是类型级别的常量，用 `Type::CONST` 访问。
2. 声明只给出名称和类型；
3. 定义**提供具体值**。
	- **标识符（identifier）**：常量在访问时使用的名称，例如 `Circle::PI` 中的 `PI`。
    - **类型（type）**：定义常量时必须匹配的类型，例如 `f64` 或 `u32`。
4. 关联常量的值只有在被访问时才会计算（类似惰性求值），不会在类型定义时就计算。
	1. 提高编译效率，不访问的常量不会消耗计算资源。
5. 如果常量定义中涉及泛型参数，编译器会先生成单态化（monomorphization，即为具体类型生成专门实现），然后再计算常量值。
    - 泛型关联常量在不同类型实例化时可能有不同值，因此需要先单态化再求值。

## 2. 背景/出处  
- 来源：https://doc.rust-lang.org/reference/items/implementations.html#inherent-implementations
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  

### Associated Constants 逐句解析

#### 1. “Associated constants are constants associated with a type.”
- **解释**：关联常量是与某个类型（`struct`、`enum` 或 `trait`）绑定的常量，而不是独立存在的全局常量。
- **意义**：它们可以通过类型路径直接访问，例如 `MyType::CONST_VALUE`，强调类型的上下文，而非全局命名空间。
- **示例**：

```rust
struct Circle;
impl Circle {
    const PI: f64 = 3.14159;
}
fn main() {
    println!("{}", Circle::PI); // 访问关联常量
}
```

---

#### 2. “An associated constant declaration declares a signature for associated constant definitions. It is written as const, then an identifier, then :, then a type, finished by a ;.”

- **解释**：声明关联常量时，只给出名称和类型，不提供具体值。这类似于函数的签名，只是用于常量。
- **结构**：
```rust
const NAME: Type;
```

- **示例**：
```rust
trait Shape {
    const SIDES: u32; // 声明，但没有定义具体值
}
```

- **作用**：允许不同类型实现这个 trait 时，为该常量提供具体值。
    

---

#### 3. “The identifier is the name of the constant used in the path. The type is the type that the definition has to implement.”

- **解释**：
    - **标识符（identifier）**：常量在访问时使用的名称，例如 `Circle::PI` 中的 `PI`。
    - **类型（type）**：定义常量时必须匹配的类型，例如 `f64` 或 `u32`。
- **意义**：保证类型安全，编译器在引用时可以检查类型是否一致。

---

#### 4. “An associated constant definition defines a constant associated with a type. It is written the same as a constant item.”

- **解释**：定义关联常量时要给出具体值，它的写法与普通 `const` 相同。
- **示例**：

```rust
impl Shape for Square {
    const SIDES: u32 = 4; // 这里定义了具体值
}
```

- **区别**：声明只是签名，定义提供具体数值。

---

#### 5. “Associated constant definitions undergo constant evaluation only when referenced. Further, definitions that include generic parameters are evaluated after monomorphization.”

- **解释第一句**：关联常量的值只有在被访问时才会计算（类似惰性求值），不会在类型定义时就计算。
- **解释第二句**：如果常量定义中涉及泛型参数，编译器会先生成单态化（monomorphization，即为具体类型生成专门实现），然后再计算常量值。
- **意义**：
    - 提高编译效率，**不访问的常量不会消耗计算资源**。
    - 泛型关联常量在不同类型实例化时可能有不同值，因此需要先单态化再求值。
- **示例**：

```rust
struct Wrapper<T>(T);
impl<T> Wrapper<T> {
    const ZERO: T = T::default(); // 只有 Wrapper<i32>::ZERO 被访问时才求值
}
```
### 6. 示例
```rust
struct Struct;
struct GenericStruct<const ID: i32>;

impl Struct {
    // Definition not immediately evaluated
    const PANIC: () = panic!("compile-time panic");
}

impl<const ID: i32> GenericStruct<ID> {
    // Definition not immediately evaluated
    const NON_ZERO: () = if ID == 0 {
        panic!("contradiction")
    };
}

fn main() {
    // Referencing Struct::PANIC causes compilation error
    let _ = Struct::PANIC;

    // Fine, ID is not 0
    let _ = GenericStruct::<1>::NON_ZERO;

    // Compilation error from evaluating NON_ZERO with ID=0
    let _ = GenericStruct::<0>::NON_ZERO;
}
```
### 官方实例
```rust
trait ConstantId {
    const ID: i32;
}

struct Struct;

impl ConstantId for Struct {
    const ID: i32 = 1;
}

fn main() {
    assert_eq!(1, Struct::ID);
}
```
Using default values:  使用默认值：
```rust
trait ConstantIdDefault {
    const ID: i32 = 1;
}

struct Struct;
struct OtherStruct;

impl ConstantIdDefault for Struct {}

impl ConstantIdDefault for OtherStruct {
    const ID: i32 = 5;
}

fn main() {
    assert_eq!(1, Struct::ID);
    assert_eq!(5, OtherStruct::ID);
}
```


## 4. 与其他卡片的关联  
- 前置卡片：
	- [Rust-Associated Items (关联项目)-基本概念](Rust-Associated%20Items%20(关联项目)-基本概念.md)
	- [Rust-Associated Items (关联项目)-关联类型](Rust-Associated%20Items%20(关联项目)-关联类型.md)
- 后续卡片：
	- [Rust-Associated Items-关联类型-关联类型不能在固有实现中定义的原因](Rust-Associated%20Items-关联类型-关联类型不能在固有实现中定义的原因.md)
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
- [ ] 深入阅读 xxx  
- [ ] 验证这个观点的边界条件  
