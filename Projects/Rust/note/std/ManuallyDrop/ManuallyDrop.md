```rust

pub struct ManuallyDrop<T>where
    T: ?Sized,{ /* private fields */ }

```
use std::mem::ManuallyDrop;
- 一个包装器，用于**禁止编译器自动调用 `T` 的析构函数**。此包装器为 0-cost。
	-  编译器不会插入额外的字段或 padding；
	- 在生成代码时，LLVM 会将 `ManuallyDrop<T>` 视作普通的 `T`；
	- 所有访问都可被折叠为裸访问（zero-overhead）；
-  不引入 drop glue（编译器生成的析构器）
	- 你必须手动执行 `ManuallyDrop::drop(&mut x)`，而且该方法是 **unsafe**，表示你必须 **确信它只被析构一次**。
- `ManuallyDrop<T>` 保证具有与 `T` 的布局优化，并且受与 `T` 相同的布局优化的约束。因此，它对编译器对其内容所做的假设_没有影响_ 。例如 `，使用` [`mem：：zeroed`](https://doc.rust-lang.org/std/mem/fn.zeroed.html "fn std::mem::zeroed") 是未定义的行为。如果您需要处理未初始化的数据，请改用 [`MaybeUninit<T>`](https://doc.rust-lang.org/std/mem/union.MaybeUninit.html "union std::mem::MaybeUninit")。
- 请注意，访问 `ManuallyDrop<T>` 中的值是安全的。这意味着其内容已被丢弃的 `ManuallyDrop<T>` 不得通过公共安全 API 公开。相应地，`ManuallyDrop：:d rop` 是不安全的。
#### 作用：
**控制对象的析构行为**，防止 Rust 自动在作用域结束时调用 `drop`。
- 在 `LazyLock` 中，`f` 被手动管理，防止在初始化前提早析构；
- 初始化后，`f` 会被安全地忘掉，然后将 `value` 写入；
- 整个过程避免了不安全的双重 drop。

### 所有访问都是“裸访问”（zero-overhead access）
`ManuallyDrop<T>` 实现了如下访问方法：
```rust
impl<T> ManuallyDrop<T> {
    pub fn into_inner(slot: ManuallyDrop<T>) -> T
    pub fn as_ptr(&self) -> *const T
    pub fn as_mut_ptr(&mut self) -> *mut T
}

```
- `.into_inner()` 是一个 **移动语义的解包**，不调用 `drop`；
- `.as_ptr()` / `.as_mut_ptr()` 允许你对其进行原始指针访问；
- 全部是 **透明、零开销的转发**。
    
这些操作编译器不会做任何额外检查，因为：
- 编译期就知道 `ManuallyDrop<T>` 没有 drop glue；
- 所以编译器优化器可以**直接 inline 展开这些函数调用为裸指针偏移和 load/store**。
#### 🔹 “**inline 展开**函数调用” 是什么？
这是指 **函数在编译过程中被展开到调用处，而不是生成真实的函数调用指令（如 `call`）**。
- 编译器会把函数体直接复制到调用点；
- 避免函数调用的跳转开销（函数栈帧、返回地址等）；
- 对编译器优化器友好（更容易进一步分析和折叠）；
- 尤其对泛型、透明包装器等“小函数”非常常见。
#### 🔹 “**裸指针偏移**” 和 “**load/store**” 指的是？
当你访问 `ManuallyDrop<T>` 中的值时，它的内存布局与 `T` 完全一致，因此可以被视为直接访问一个裸 `*mut T` 或 `*const T` 指针。例如：

```rust
use std::mem::ManuallyDrop;

fn example() {
    let mut x = ManuallyDrop::new(123u32);

    unsafe {
        let p: *mut u32 = ManuallyDrop::as_mut_ptr(&mut x);
        *p = 456;
    }
}
****

```
这一段在 LLVM 层基本等价于：
```rust
%x = alloca i32
store i32 123, i32* %x
store i32 456, i32* %x

```
`.as_mut_ptr()` 实际是一个 `&mut self as *mut T`，没有运行时开销；
编译器视作**直接裸指针偏移 + 原始 load/store 指令**。

#### 为什么可以做到这一点

- **`[repr(transparent)]`**
    - 保证 `ManuallyDrop<T>` 的布局 == `T`；
    - 编译器可以直接“transmute”成裸指针；
- **无 Drop 实现**
    - 编译器知道不需要生成 drop glue；
    - 不需要考虑“还没 drop 就访问”的问题；
- **泛型内联 + monomorphization**
    - `ManuallyDrop::as_mut_ptr` 是泛型函数；
    - 编译器会针对每个具体类型 `T` 展开 monomorphized 实现；
    - 所以调用点直接被 inline 替换为裸访问逻辑。
```rust
use std::mem::ManuallyDrop;

#[inline(always)]
fn example() {
    let mut x = ManuallyDrop::new(42u32);
    unsafe {
        *ManuallyDrop::as_mut_ptr(&mut x) = 100;
    }
}


```
使用如下命令输出 LLVM IR：
```
cargo rustc --release -- --emit=llvm-ir

```
你会看到类似 IR：
```
%x = alloca i32, align 4
store i32 42, i32* %x, align 4
store i32 100, i32* %x, align 4

```
- 并没有调用任何函数（如 `ManuallyDrop::as_mut_ptr`）；
- 所有东西都被**编译器优化为裸指针读写**；
- `.as_mut_ptr()` 被展开为 `&mut self as *mut T`；
- 编译器认定 `ManuallyDrop<T>` 就是 `T`，所以 IR 层没有任何抽象成本。


可以**直接 inline 展开这些函数调用为裸指针偏移和 load/store**。
Rust 编译器会将 `ManuallyDrop` 的访问方法（如 `into_inner`、`as_ptr`）进行 **monomorphization + inline 展开**，其结果在 LLVM IR 层被等效转化为 **原始内存地址的偏移读取或写入（load/store）**，完全不会产生运行时函数调用或中间封装逻辑，真正实现 **zero-cost abstraction**。
### 常见方法的使用



### 编译之后零成本抽象和T的关系
这个现象正体现了 Rust 所强调的设计哲学之一：**Zero-cost Abstractions** —— 使用高级抽象语义时，**只在编译期引入约束和安全性检查，不在运行期引入开销**。我们可以进一步拆解你的结论：
#### 1. 编译后完全等价：`ManuallyDrop<T>` == `T`
- 编译器在生成代码时，会把 `ManuallyDrop<T>` 直接当作 `T` 本身对待。
- 所有字段访问、赋值、内存布局、指针偏移等操作都不会插入额外逻辑。
- 甚至在 LLVM 层，它根本看不出你用了一个“包装器”。
#### 2. 写代码时的目的：**表达语义、避免自动 drop**
你之所以要在代码中显式写 `ManuallyDrop<T>`，是因为：
- **告诉编译器：你不希望自动析构 T**；
- **启用 unsafe 控制：你自己负责 drop 的时机和逻辑**；
- 编译器借此在作用域结束时 **不插入 drop glue**；
- 静态检查系统也知道“这里不要自动 drop”，比如：可以 move 出值、不调用析构、不发生双 drop 等。
🔸 普通写法（会自动 drop）：
```rust
fn example() {
    let s = String::from("hello");
} // <- s 在这里自动 drop

```
🔸 使用 `ManuallyDrop`（你来手动 drop）：
```rust
use std::mem::ManuallyDrop;

fn example() {
    let mut s = ManuallyDrop::new(String::from("hello"));
    
    unsafe {
        // 手动 drop
        ManuallyDrop::drop(&mut s);
    }
} // <- 编译器不会自动 drop s，避免 double-drop

```

#### 为何“写法有区别，运行等价”？

| 特性    | 普通 `T`            | `ManuallyDrop<T>`           |
| ----- | ----------------- | --------------------------- |
| 语法差异  | 普通构造、作用域析构        | 必须 `.new()`，可 `unsafe` drop |
| 编译时行为 | 编译器自动插入 drop glue | 编译器不插入 drop glue            |
| 运行时行为 | 自动析构              | 你手动控制析构                     |
| 内存布局  | 等同于 `T`           | 完全等同于 `T`                   |
| 性能    | 无额外成本             | 也无成本，zero-cost              |
#### drop glue

`drop glue` 是 Rust 编译器自动生成的析构代码，用来在变量生命周期结束时调用它的 `Drop` 实现（如果有），并递归释放所有字段。
##### 为什么叫 “glue”？
“Glue” 直译是“胶水”，在编译器术语中指的是**把多个行为粘合起来的一段自动生成代码**。
- 自动在作用域末尾生成；
- 自动调用值的 `Drop::drop()` 方法；
- 自动对结构体字段做递归 drop；
- 保证所有资源都在“正确的顺序”释放。
```rust
struct MyData {
    name: String,
    buffer: Vec<u8>,
}

fn example() {
    let data = MyData {
        name: String::from("Alice"),
        buffer: vec![1, 2, 3],
    };
} // <-- 这里自动插入 drop glue

```
编译器隐含的 drop glue 相当于：
```rust
impl Drop for MyData {
    fn drop(&mut self) {
        // drop glue 会自动生成类似逻辑：
        drop(self.buffer);
        drop(self.name);
    }
}

```
即使你没有写 `Drop` trait，编译器也会生成这段 “drop glue”。
##### ⚙️ drop glue 的行为特征

|特性|说明|
|---|---|
|自动生成|你写不写 `Drop` 都会生成|
|递归析构|会深入结构体/枚举的字段|
|顺序一致|按照字段定义顺序，从后往前 drop|
|编译期决定|Rust 在编译时分析生命周期并插入|
|可被优化|如果没有资源管理需求，可以被省略|
