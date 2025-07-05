### 开发构建配置
Rust 编译器通过 **Profile** 来控制编译时的优化等级、调试信息、panic 策略等参数
```rust
[profile.dev]
opt-level = 0       # 关闭优化，编译快，调试友好
debug = true        # 生成调试符号
panic = "abort"     # panic 时直接中止程序
overflow-checks = true  # 开启整数溢出检查
```
### 设置panic栈展开配置

```
[profile.dev]
panic = "abort"

[profile.release]
panic = "abort"

```
这些选项能将 **dev 配置**（dev profile）和 **release 配置**（release profile）的 panic 策略设为 `abort`。`dev` 配置适用于 `cargo build`，而 `release` 配置适用于 `cargo build --release`。现在编译器应该不再要求我们提供 `eh_personality` 语言项实现。