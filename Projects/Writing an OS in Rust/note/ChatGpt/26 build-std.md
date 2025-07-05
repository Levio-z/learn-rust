- 通常状况下，`core` crate以**预编译库**（precompiled library）的形式与 Rust 编译器一同发布——这时，`core` crate只对支持的宿主系统有效，而对我们自定义的目标系统无效。**如果我们想为其它系统编译代码，我们需要为这些系统重新编译整个 `core` crate。**
#### `build-std` 选项
此时就到了cargo中 [`build-std` 特性](https://doc.rust-lang.org/nightly/cargo/reference/unstable.html#build-std) 登场的时刻，该特性允许你按照自己的需要重编译 `core` 等标准crate，而不需要使用Rust安装程序内置的预编译版本。 但是该特性是全新的功能，到目前为止尚未完全完成，所以它被标记为 “unstable” 且仅被允许在 [Nightly Rust 编译器](https://os.phil-opp.com/zh-CN/minimal-rust-kernel/#an-zhuang-nightly-rust) 环境下调用。
要启用该特性，你需要创建一个 [cargo 配置](https://doc.rust-lang.org/cargo/reference/config.html) 文件，即 `.cargo/config.toml`，并写入以下语句：

```toml
# in .cargo/config.toml

[unstable]
build-std = ["core", "compiler_builtins"]
```

该配置会告知cargo需要重新编译 `core` 和 `compiler_builtins` 这两个crate，其中 `compiler_builtins` 是 `core` 的必要依赖。 另外重编译需要提供源码，我们可以使用 `rustup component add rust-src` 命令来下载它们。

执行 `cargo build` 之后， `core`、和 `compiler_builtins` crate被重新编译了。