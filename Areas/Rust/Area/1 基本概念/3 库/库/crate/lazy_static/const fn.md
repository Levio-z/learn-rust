在 Rust 中，`const fn` 表示 **可以在编译期执行的函数**。

> 换句话说：你可以在 `const` 或 `static` 上下文中调用这个函数，并直接获得一个**常量结果**，而不需要运行时计算。

示例：普通函数 vs `const fn`
```rust
fn square(x: i32) -> i32 {
    x * x
}

// ❌ 错误：不能在 const 中调用普通函数
const VAL: i32 = square(5);

```
改成 `const fn`：
```rust
const fn square(x: i32) -> i32 {
    x * x
}

const VAL: i32 = square(5);  // ✅ 编译期求值，OK

```
### 使用场景
#### 1. 初始化 `const` 或 `static` 变量
#### 2. 支持泛型常量（const generics）
```rust
struct Buffer<const N: usize> {
    data: [u8; N],
}

const fn default_size() -> usize { 128 }

type DefaultBuf = Buffer<{ default_size() }>;

```
#### 3. 编译期构造复杂结构（比如查表、构造 LUT、哈希种子）
由于编译器必须保证编译期运行的代码是确定性、无副作用的，`const fn` 的能力是逐步放开的。目前支持：
- 运算、条件分支、循环（`while` / `for`）
- 枚举匹配 (`match`)
- `&mut` 可变引用（带限制）
- 泛型函数（可带 `const` 参数）
- `Result` / `Option` 等控制流
- **从 Rust 1.61 起：支持 `if let`、`while let` 等模式匹配**
不允许（或受限）：
- `unsafe` 块（部分支持，需标记）
- 动态分配（堆上分配）
- IO、线程、FFI 等运行时代码
- 一个典型例子：`const fn` 构造静态查找表
```rust
const fn fib(n: usize) -> usize {
    if n == 0 { 0 } else if n == 1 { 1 } else { fib(n-1) + fib(n-2) }
}

const FIB_TABLE: [usize; 10] = {
    let mut arr = [0; 10];
    let mut i = 0;
    while i < 10 {
        arr[i] = fib(i);
        i += 1;
    }
    arr
};


```
你可以在 **编译期** 构造这个数组，**0 运行时开销**。
🔬 和 `const` / `static` 的关系

|术语|含义|
|---|---|
|`const`|定义一个编译期求值的常量（不可修改、无地址）|
|`static`|定义一个有 `'static` 生命周期的全局变量（可有地址）|
|`const fn`|可以在 `const` 或 `static` 上下文中使用的函数|