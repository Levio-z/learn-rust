## 1. Rust 的 crate 命名解析模型

Rust 编译系统（Cargo + rustc）中：

- **每个 `[lib]` 或 `[[bin]]` 定义的入口文件（lib.rs / main.rs / bin/xxx.rs）都是一个 crate**
    
- crate 有自己独立的 `crate` 作用域
    
- 在 bin crate 中，`crate::` 指向的是自己的 main 文件模块，而不是 lib crate
    
- 所以 bin crate 要访问 lib crate 的内容，需要通过“包名”来引用（就像引用外部库一样）

### 2. crate 与 bin 同名不会冲突的原因
`[lib]` 默认为包名

|项目结构|含义|
|---|---|
|`vistor-file-handler`（Cargo.toml）|package 名|
|`[lib]` name = `"vistor-file-handler"`|lib crate，导出的库模块|
|`[[bin]]` name = `"vistor-file-handler"`|bin crate，最终编译为 `vistor-file-handler` 可执行文件|

Rust 的 crate 系统并不会用名字去“区分作用域”，它用 **文件路径 + crate 类型** 管理每个 crate：
- `lib.rs` 是 crate root for lib
- `bin/bin.rs` 是 crate root for bin
- 它们是 **完全隔离的 crate**
 所以即使同名也不会冲突——**你在 bin.rs 中访问的是 lib crate，不是自己这个 bin crate**
### 3. 在 bin.rs 中能访问 lib.rs 的关键

```rust
// Cargo.toml
[package]
name = "vistor-file-handler"

// bin/bin.rs
use vistor_file_handler::SomeType; // ✅ 正确！Cargo 会隐式链接 lib crate

```
等价于
```rust
extern crate vistor_file_handler;
use vistor_file_handler::SomeType;

```

### 4. Rust 是如何区分 crate 的？
查看：
```rust
cargo metadata --format-version 1 | jq '.packages[].targets[] | {name, kind, crate_types}'

```
按照包名筛选
```rust
cargo metadata --format-version 1 \
  | jq '.packages[] | select(.name == "vistor-file-handler") | .targets[] | {name, kind, crate_types}'

```