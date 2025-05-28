### trait 定义和作用
`From` 是 Rust 标准库提供的一个通用转换 trait，定义在 `std::convert` 模块：

`pub trait From<T>: Sized {     fn from(T) -> Self; }`

- 它表示“从类型 `T` 转换成当前类型”。
- 一旦实现了 `From<T>`，就可以用 `T::from(value)` 来得到当前类型的实例。
- 同时，`From` 实现会自动为 `Into` trait 提供实现，反向转换也可用。
#### 使用
- 显式调用：
	- `let my_err: MyError = MyError::from(io_err); `
	- `let my_err: MyError = From::from(io_err);`
- 隐式调用
	- `fn from<T, U>(t: T) -> U where U: From<T>`
	- 但是你注意到返回类型 `U` 是由上下文推断的。
	- 案例，就是？底层就是
		```rust
		fn foo() -> Result<(), MyError> {
		    let err: io::Error = ...;
		    return Err(From::from(err));
		}
		```
### details

#### **Rust 标准库中确实有为所有类型实现 `From` 到自身的默认实现**
- 这叫做“Identity conversion”（身份转换）。
```rust
impl<T> From<T> for T {
    fn from(t: T) -> T {
        t
    }
}
```
### Into
#### 1. 简单示例：定义一个错误类型，实现 `From`
```rust
use std::io;

#[derive(Debug)]
enum MyError {
    IoError(io::Error),
    Other,
}

impl From<io::Error> for MyError {
    fn from(err: io::Error) -> Self {
        MyError::IoError(err)
    }
}
```
这时，`MyError` 自动实现了：
```rust
impl Into<MyError> for io::Error {
    fn into(self) -> MyError {
        MyError::from(self)
    }
}
```
#### 2. `Into` 的使用示例
```rust
fn convert_error(err: io::Error) -> MyError {
    // 显式调用 Into
    let my_err: MyError = err.into();
    my_err
}

fn main() {
    let io_err = io::Error::new(io::ErrorKind::Other, "something went wrong");
    let my_err = convert_error(io_err);
    println!("{:?}", my_err);
}

```
#### 3. 也可以用 `Into` 在泛型函数中
```rust
fn print_error<E: Into<MyError>>(err: E) {
    let my_err: MyError = err.into();
    println!("Error: {:?}", my_err);
}

fn main() {
    let io_err = io::Error::new(io::ErrorKind::Other, "fail");
    print_error(io_err);  // 这里 io::Error 自动通过 Into 转为 MyError

    let my_err = MyError::Other;
    print_error(my_err);  // MyError 本身也能用 Into<MyError>，identity conversion
}


```