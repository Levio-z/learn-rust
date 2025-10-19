- 整数类型默认i32，使用 `isize` 或 `usize` 的主要情况是为某种集合编制索引时。

| Length  长度    | Signed  签名 | Unsigned  未签名 |
| ------------- | ---------- | ------------- |
| 8-bit  8位     | `i8`       | `u8`          |
| 16-bit  16位   | `i16`      | `u16`         |
| 32-bit  32位   | `i32`      | `u32`         |
| 64-bit  64位   | `i64`      | `u64`         |
| 128-bit  128位 | `i128`     | `u128`        |
| arch  拱       | `isize`    | `usize`       |
- 当符号重要时，数字会显示为加号或减号；
```rust
println!("{:+}", 42);   // 输出：+42
```
- 但是，当可以安全地假设该数字为正数时，它会不显示任何符号。
```rust
println!("{}", 42);   // 输出：42 （没有 +，因为正号可省略）
println!("{}", -42);  // 输出：-42 （负号必须显示）
```
- 范围：
	- 每个带符号的变体可以存储从 $-(2^{n-1})$ 到 $2^{n-1} -1$（含）的数字，其中`n`是变体使用的位数。
		- 所以一个 `i8`可以存储从$-(2^{7})$ 到 $2^{7} -1$ 的数字，等于 -128 到 127。
	- 无符号变体可以存储从 0 到 $2^{n} -1$的数字
		- 因此`u8`可以存储从 0 到 2 8 - 1 的数字，等于 0 到 255。
- 此外， `isize`和`usize`类型取决于程序运行所在计算机的体系结构，在表中表示为“arch”：如果使用 64 位体系结构，则为 64 位；如果使用 64 位体系结构，则为 32 位在 32 位架构上。

### 表示

- 类型后缀：请注意，可以是多个数字类型的数字文字允许使用类型后缀（例如`57u8` ）来指定类型。
- **视觉分隔符**`_`：数字文字还可以使用`_`作为视觉分隔符，以使数字更易于阅读，例如`1_000` ，它的值与您指定的`1000`相同。

|Number literals  数字文字|Example  例|
|---|---|
|Decimal  十进制|`98_222`|
|Hex  十六进制|`0xff`|
|Octal  八进制|`0o77`|
|Binary  二元的|`0b1111_0000`|
|Byte (`u8` only)  字节（仅限 `u8`）|`b'A'`|
### 整数溢出
- 假设您有一个类型`为 u8` 的变量，可以保存 0 到 255 之间的值。如果尝试将变量更改为该范围之外的值（例如 256），则会发生整数溢出
	- 调试模式
		- 运行时崩溃
	- --release
		- 执行二进制补码换行
			- 简而言之，大于类型可以保存的最大值的值“环绕”到类型可以保存的最小值。对于 `u8`，值 256 变为 0，值 257 变为 1，依此类推。程序不会恐慌，但变量的值可能不是你期望的。依赖整数溢出的包装行为被视为错误。
#### 显式处理溢出的可能性
- **`wrapping_*`**：发生溢出时按二进制补码环绕（wrap around）。
- **`checked_*`**：发生溢出时返回 `None`。
- **`overflowing_*`**：返回 `(结果值, 是否溢出)`。
- **`saturating_*`**：发生溢出时固定到最小值或最大值（饱和运算）。

```rust
// 允许在编译期写出会溢出的算术表达式，不把它当错误。
#![allow(arithmetic_overflow)]
fn main() {
    let a: u8 = 250;
    let b: u8 = 10;

    println!("=== wrapping_* 示例 ===");
    // 250 + 10 = 260，本应溢出
    
    println!("a+b: {}", a+b);
    println!("wrapping_add: {}", a.wrapping_add(b)); // 4 (260 % 256)
    println!("wrapping_sub: {}", 5u8.wrapping_sub(10)); // 251 (向下溢出)

    println!("\n=== checked_* 示例 ===");
    match a.checked_add(b) {
        Some(v) => println!("checked_add: {}", v),
        None => println!("checked_add: overflow -> None"),
    }
    match 5u8.checked_sub(10) {
        Some(v) => println!("checked_sub: {}", v),
        None => println!("checked_sub: overflow -> None"),
    }

    println!("\n=== overflowing_* 示例 ===");
    let (v1, of1) = a.overflowing_add(b);
    println!("overflowing_add: {} (溢出? {})", v1, of1);

    let (v2, of2) = 5u8.overflowing_sub(10);
    println!("overflowing_sub: {} (溢出? {})", v2, of2);

    println!("\n=== saturating_* 示例 ===");
    println!("saturating_add: {}", a.saturating_add(b)); // 255 (饱和在最大值)
    println!("saturating_sub: {}", 5u8.saturating_sub(10)); // 0 (饱和在最小值)
}

```
结果：
```shell

=== wrapping_* 示例 ===
a+b: 4
wrapping_add: 4
wrapping_sub: 251

=== checked_* 示例 ===
checked_add: overflow -> None
checked_sub: overflow -> None

=== overflowing_* 示例 ===
overflowing_add: 4 (溢出? true)
overflowing_sub: 251 (溢出? true)

=== saturating_* 示例 ===
saturating_add: 255
saturating_sub: 0

```