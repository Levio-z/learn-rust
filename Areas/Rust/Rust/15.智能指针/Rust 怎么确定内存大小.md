```rust
enum List {
    Cons(i32, Box<List>),
    Nil,
}
```
从**编译器内存布局角度**严谨地推断出它的内存大小。
枚举的大小由以下因素共同决定：
1. **每个变体的大小**
2. **最大变体的大小**
3. **加上一个“tag”（标签）字节**，用于标记当前是哪个变体
### `List` 枚举的两个变体
#### `Cons(i32, Box<List>)`
- `i32`：固定 4 字节
- `Box<List>`：在 64 位系统上是 8 字节的裸指针
- 合计大小：`4 + 8 = 12` 字节
- 但因为 `Box` 紧跟在 `i32` 后，为满足 8 字节对齐，**可能产生填充（padding）**
    - `i32`：4字节
    - padding：4字节（为了对齐后面的指针）
    - `Box<List>`：8字节
实际大小：**16 字节**
#### `Nil`
- 什么都不存储，理论上是 `0` 字节
- 但枚举需要区分当前是 `Cons` 还是 `Nil`，因此需要一个 **tag（变体标记）**
- Rust 会为整个枚举分配一个 tag，足够区分所有变体（这里只有两个，所以 `1` 字节就够）
#### 最终大小
- 枚举类型会为 **所有变体分配同样大小**的内存。
- 编译器会选择**最大的变体**所需的内存大小作为整个 enum 的大小。
	- 从**概念上**来看，`Nil` 自己 **只代表一个空变体，不存储任何数据**。但从**内存布局角度**，因为 `Nil` 是 `List` 枚举的一种变体，而 Rust 枚举的大小是 **统一的**。
结论：
- 最大变体是 `Cons`：16 字节，加上 tag（Rust 可能将 tag 嵌入 padding 中，以节省空间）
- **在实践中，Rust 会布局得足够紧凑，不会额外添加 tag 字节**，而是复用 padding：
>  **`List` 的总大小就是 16 字节**
#### 验证
```rust
use std::mem::size_of;

enum List {
    Cons(i32, Box<List>),
    Nil,
}

fn main() {
    println!("Size of List: {}", size_of::<List>()); // 输出：16
}

```
最大变体是 `Cons`：16 字节
`i32`：4字节
    - padding：4字节（为了对齐后面的指针）
    - `Box<List>`：8字节
加上 tag（Rust 可能将 tag 嵌入 padding 中，以节省空间）

```rust
use std::mem::size_of;

enum BoxedEnum {
    A(Box<u8>),
    B(Box<u8>),
}

fn main() {
    println!("Size of BoxedEnum: {}", size_of::<BoxedEnum>()); // 输出: 9
}


```

| 字段                | 大小         |
| ----------------- | ---------- |
| `Box<u8>`         | 8 字节       |
| discriminant（tag） | 1 字节       |
| **总大小**           | **9 字节** ✅ |
**输出为 16 是正确且预期的**，虽然我们逻辑上只需要 `Box<u8>`（8 字节）+ 1 字节 tag，**但由于对齐（alignment）规则，Rust 将总大小对齐到 8 字节倍数，最终为 16 字节**。
### 对齐&对齐
填充（padding）的设计**并不是保证每个字段都按照最大字段对齐**，而是保证**每个字段都按照它自身的对齐要求对齐**，同时保证整个结构体的大小是最大字段对齐的倍数。

具体来说：

- **每个字段的地址**会被填充调整到满足该字段自身的对齐要求（比如一个 `u16` 字段对齐2字节，一个 `u64` 字段对齐8字节）。
    
- 结构体整体的起始地址必须满足结构体的对齐（即最大字段的对齐要求），以保证结构体作为整体时能正确对齐。
    
- **结构体大小（size）会被填充到最大字段对齐的倍数**，确保结构体数组中每个元素起始地址都满足最大字段对齐。
```rust
use std::mem::{align_of, size_of};

// 基础对齐：只有 u8 字段，1 字节对齐
struct Align1 {
    a: u8,
}

// 多字段，最大对齐是 u32，4 字节对齐
struct Align4 {
    a: u8,
    b: u32,
}

// 最大字段是 u64，对齐为 8 字节
struct Align8 {
    a: u8,
    b: u64,
}

// 手动指定对齐为 16 字节
#[repr(align(16))]
struct Align16 {
    a: u8,
}

// 结构体中包含一个自定义强制 32 字节对齐的字段
#[repr(align(32))]
struct Align32Field(u8);

struct Align32 {
    a: u8,
    b: Align32Field,
}

fn main() {
    println!("Align1 size: {}, align: {}", size_of::<Align1>(), align_of::<Align1>());
    println!("Align4 size: {}, align: {}", size_of::<Align4>(), align_of::<Align4>());
    println!("Align8 size: {}, align: {}", size_of::<Align8>(), align_of::<Align8>());
    println!("Align16 size: {}, align: {}", size_of::<Align16>(), align_of::<Align16>());
    println!("Align32 size: {}, align: {}", size_of::<Align32>(), align_of::<Align32>());
}

```
运行结果
```
Align1 size: 1, align: 1
Align4 size: 8, align: 4
Align8 size: 16, align: 8
Align16 size: 16, align: 16
Align32 size: 32, align: 32
```