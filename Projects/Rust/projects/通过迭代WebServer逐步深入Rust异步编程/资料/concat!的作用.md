### 基本定义
- 宏接受多个字符串字面量作为参数。
- 在编译阶段将它们直接拼接成一个新的字符串字面量。
- 生成的字符串是一个 `&'static str`，常驻程序二进制。
- **编译期拼接**，效率极高，没有运行时开销。
### 例子
```rust
let response = concat!("HTTP/1.1 200 OK\r\n", "Content-Length: 13\r\n", "\r\n", "Hello, world!");
println!("{}", response);
```
编译后相当于：
```rust
let response = "HTTP/1.1 200 OK\r\nContent-Length: 13\r\n\r\nHello, world!";

```
### 主要特点
- **编译期拼接**，效率极高，没有运行时开销。
- 只能拼接**字符串字面量**，不能拼接变量或运行时字符串。
- 生成的字符串类型是 `&'static str`，生命周期贯穿整个程序。