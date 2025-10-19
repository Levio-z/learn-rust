---
tags:
  - note
---

### Rust `std::ops::Sub` Trait 研究

#### 1. 定义与作用

`std::ops::Sub` 是 Rust 标准库中定义的用于减法运算的 **运算符重载 trait**。它允许自定义类型支持 `-` 操作符。其核心定义如下：

```rust
pub trait Sub<Rhs = Self> {
    type Output;

    fn sub(self, rhs: Rhs) -> Self::Output;
}
```

**解释：**

- `Rhs`：表示右操作数类型，默认与 `Self` 相同，可以通过泛型指定不同类型。
    
- `Output`：表示减法运算的返回类型，可以与 `Self` 不同。
    
- `fn sub(self, rhs: Rhs) -> Self::Output`：核心方法，实现 `-` 运算的逻辑。
    

**作用：**

- 支持自定义类型的减法运算。
    
- 可用于重载 `-` 运算符。
    
- 与 `Add`、`Mul`、`Div` 等算术 trait 一样，是 Rust 中算术运算符统一的抽象接口。
    

---

#### 2. 使用示例

**基础类型实现：**

```rust
let a = 10;
let b = 3;
let c = a - b; // 通过 std::ops::Sub trait
println!("{}", c); // 输出 7
```

**自定义类型实现：**

```rust
use std::ops::Sub;

#[derive(Debug)]
struct Point {
    x: i32,
    y: i32,
}

impl Sub for Point {
    type Output = Point;

    fn sub(self, other: Point) -> Point {
        Point {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

fn main() {
    let p1 = Point { x: 5, y: 7 };
    let p2 = Point { x: 2, y: 3 };
    let p3 = p1 - p2;
    println!("{:?}", p3); // Point { x: 3, y: 4 }
}
```

**不同类型减法：**

```rust
struct Millimeters(u32);
struct Meters(u32);

impl Sub<Meters> for Millimeters {
    type Output = Millimeters;

    fn sub(self, rhs: Meters) -> Millimeters {
        Millimeters(self.0 - rhs.0 * 1000)
    }
}
```

---

#### 3. 源码分析

在 Rust 标准库中，`Sub` 对基本类型是通过宏实现的，例如：

```rust
macro_rules! impl_op {
    ($Trait:ident, $method:ident, $t:ty) => {
        impl $Trait for $t {
            type Output = $t;
            fn $method(self, rhs: $t) -> $t {
                self - rhs
            }
        }
    };
}

impl_op!(Sub, sub, i32);
impl_op!(Sub, sub, f64);
```

特点：

- 对基本类型，直接使用底层 LLVM IR 的减法指令。
    
- 对自定义类型，可自定义逻辑。
    
- 支持泛型，允许跨类型操作。
    

---

#### 4. 使用场景

1. **自定义数值类型**：如向量、矩阵、单位类型等。
    
2. **坐标运算**：几何点或矢量的加减。
    
3. **金融/度量单位**：如货币或长度单位转换。
    
4. **复合数据结构**：通过 `Sub` 实现元素级别的减法。
    

---

#### 5. 扩展知识点

- **Rhs 默认值**：`Sub<Rhs = Self>` 允许右操作数类型不同，支持灵活操作。
    
- **Output 与类型转换**：可返回与 `Self` 不同的类型，实现多态减法。
    
- **组合运算 trait**：`Add`、`Sub`、`Mul`、`Div`、`Rem` 构成基础算术 trait，常与 `AddAssign`、`SubAssign` 等可变赋值 trait 配合。
    
- **泛型实现**：可以为泛型类型或 trait bounds 类型实现减法，例如 `T: Sub<Output=T>`。
    

---

### 总结

`std::ops::Sub` 是 Rust 中运算符重载体系的一部分，允许用户自定义类型实现减法运算，并可控制返回类型和右操作数类型。它在数学建模、几何计算、金融和度量单位转换等场景非常有用。

**学习方法论：**

1. 理解 trait 定义、泛型参数和关联类型。
    
2. 实践自定义类型实现 `Sub`，调试 `-` 运算符行为。
    
3. 扩展到 `Add`、`Mul`、`Div` 形成算术运算体系。
    

**推荐习题：**

1. 为 3D 向量类型实现减法。
    
2. 实现单位换算类型（如 `Cm - M`）。
    
3. 实现复合数据结构的逐元素减法。
    

**底层重点：**

- trait 泛型参数 `Rhs` 的灵活性。
    
- `Output` 类型可变实现多态。
    
- 基本类型实现与自定义类型实现的区别。
    
- 与 `AddAssign`、`SubAssign` 等 trait 组合使用的语义。
    

---

如果你希望，我可以帮你画一张 **`Sub` trait 与其他算术 trait 的关系图**，直观理解整个运算符重载体系。你希望我画吗？

## 参考
- https://rustwiki.org/zh-CN/std/ops/trait.Sub.html