### 编译查询
```rust
rustc --print target-list | grep riscv
```
### 编译设置
```rust
rustup target add riscv64gc-unknown-none-elf
```
在 `os` 目录下新建 `.cargo` 目录，并在这个目录下创建 `config` 文件，并在里面输入如下内容：
```rust
# os/.cargo/config
[build]
target = "riscv64gc-unknown-none-elf"
```