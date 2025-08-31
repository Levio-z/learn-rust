### 核心
- **自动支持格式化功能**：一旦实现了 `fmt::Write`，你就可**以使用 `write!`、`writeln!`** 这样的宏来格式化写入内容。你可以自定义如何将数据写入到某个目标（例如：日志文件、缓冲区、网络流等）。






### 底层原理
write!(vga_buffer::WRITER.lock(), "\n{} {}", 42, 1.337).unwrap(); 为什么这里能自动处理其他类型，其实只实现了打印&str的方法？
##### 1. **`write!` 宏**

`write!` 宏用于将格式化数据写入实现了 `std::fmt::Write` trait 的对象中，类似于 `println!`，但是它可以选择性地输出到其他目标，而不仅限于标准输出（例如 `stdout`）。`write!` **会接受一个实现了 `Write` 或 `fmt::Write` trait 的类型，并且支持格式化字符串**。

- `vga_buffer::WRITER.lock()` 是一个实现了 `Write` 或 `fmt::Write` 的对象（假设是 `Mutex` 中的 `VGAWriter` 类型）。
    
- `"\n{} {}"` 是一个格式化字符串，告诉 `write!` 如何格式化其后面的参数。
    

##### 2. **`std::fmt::Display` trait**

Rust 的标准库中，**`write!` 宏会基于传入类型实现的 `std::fmt::Display` trait 来处理打印格式**。`Display` 是专门用于格式化打印的 trait，提供了如何将对象格式化为人类可读的字符串的规则。

- **`i32`（数字类型）** 实现了 `Display`，因此可以使用 `{}` 来格式化打印。
    
- **`f64`（浮点数类型）** 同样实现了 `Display`，因此也能使用 `{}` 格式化打印。
    

这些类型（`i32`, `f64` 等）通过实现 `Display` trait 来支持格式化打印。具体来说，当你传入 `42` 和 `1.337` 时，`write!` 会检查这些值的类型，并根据相应的 trait 来调用格式化方法。

##### 3. **`write!` 如何自动处理不同类型**

当 `write!` 宏执行时，它会查找相应的类型实现的 `fmt::Display` 或 `fmt::Debug` trait。例如：

- **对于 `42`，它是一个 `i32` 类型，`i32` 实现了 `Display` trait，因此 `write!` 会自动将 `42` 转换为字符串 `"42"` 并插入到格式化字符串中**。
    
- 对于 `1.337`，它是一个 `f64` 类型，`f64` 同样实现了 `Display` trait，因此它会被格式化为 `"1.337"`。
    

这些类型都实现了 `fmt::Display`，因此 `write!` 可以自动将它们转换为字符串并进行格式化。

##### 4. **锁的作用**

`vga_buffer::WRITER.lock()` 在这里是获取 `WRITER`（假设是 `Mutex<SomeType>`）的锁，以确保线程安全地访问共享资源。这里的 `SomeType` 类型应该实现了 `std::fmt::Write` trait，才可以通过 `write!` 宏进行写入。

`fmt::Write` trait 与 `std::fmt::Write` 类似，只不过它更注重于类型的格式化输出。例如，在 VGA 或其他类似设备的底层输出时，你会使用一个实现了 `fmt::Write` 的类型。

##### 总结

`write!` 宏能自动处理其他类型（如 `i32` 和 `f64`），是因为：

1. **`std::fmt::Display`** trait：当你传递给 `write!` 宏的参数类型实现了 `Display` trait 时，Rust 会自动调用该类型的 `fmt` 方法，将其转换为合适的格式。
    
2. **自动格式化**：对于数字类型、字符串、浮点数等常见类型，Rust 已经为它们实现了 `Display` trait，因此你可以直接使用 `{}` 占位符进行格式化。
    
3. **`write!` 宏的实现**：`write!` 会根据格式化字符串的要求，将传入的每个参数通过其相应的 `fmt::Display` 实现格式化并写入目标对象（在这个例子中是 `vga_buffer::WRITER`）。
