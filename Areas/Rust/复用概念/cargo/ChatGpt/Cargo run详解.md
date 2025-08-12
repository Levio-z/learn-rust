### 📁 `build/` 目录

```yaml
drwxrwxr-x. 2 jimb jimb   4096 Sep 22 21:37 build
```

> Cargo 用于构建过程中的中间构件（如宏生成器、构建脚本的产物等）的缓存目录。
> 
> -   自动生成，无需手动管理。
>     

---

### 📁 `deps/` 目录

```yaml
drwxrwxr-x. 2 jimb jimb   4096 Sep 22 21:37 deps
```

> 存放依赖项编译后的 `.rlib` / `.rmeta` 文件或已链接 `.d` 依赖描述文件，便于主程序链接。
> 
> -   例如你依赖的 crates（如 `log`, `serde`）的编译产物就在此。
>     

---

### 📁 `examples/` 目录

```yaml
drwxrwxr-x. 2 jimb jimb   4096 Sep 22 21:37 examples
```

> 如果你项目中有 `examples/` 文件夹（用于 `cargo run --example xxx`），对应的可执行文件会输出在这里。

---

### ✅ `hello` 可执行文件

```diff
-rwxrwxr-x. 1 jimb jimb 576632 Sep 22 21:37 hello
```

> 编译得到的主程序执行文件（如 `main.rs` 或 `bin/hello.rs`）：

-   文件大小约 576KB
    
-   可执行权限 `rwxrwxr-x`
    
-   编译命令通常为：
    
    ```bash
    cargo build
    ```
    
-   执行命令：
    
    ```bash
    ../target/debug/hello
    ```
    
-   输出：
    
    ```text
    Hello, world!
    ```
    

---

### 📄 `hello.d` 依赖描述文件

```diff
-rw-rw-r--. 1 jimb jimb    198 Sep 22 21:37 hello.d
```

> 编译器生成的 *Makefile-style* 依赖描述文件（用于增量编译判断是否需要重新构建）。  
> 记录了构建 `hello` 可执行文件时所依赖的源文件列表。

- **支持 Makefile 风格的增量编译机制**，让构建系统能判断哪些文件改变了，从而决定是否需要重新编译目标产物。
- 
#### 机制
- 当你执行 `cargo build` 时：
    
    - Rust 会调用 `rustc` 进行编译。
        
    - 编译器会分析 **源代码依赖关系**（例如 `mod`、`use`）。
        
    - 然后输出 `.d` 文件来记录所有相关的源文件路径。
        
- 这个 `.d` 文件可以被 `make` 或其他构建系统（如 Bazel、Ninja）用于 **构建缓存与增量编译决策**。
    
- 虽然 Cargo 本身并不使用 `hello.d`（它使用 `Cargo.lock`、`fingerprint` 和元数据进行更高层次的依赖管理），但 `hello.d` 是由 `rustc` 按照传统习惯保留的产物。
---

### 📁 `incremental/` 目录

```
drwxrwxr-x. 2 jimb jimb     68 Sep 22 21:37 incremental
```

> Rust 的 **增量编译缓存目录**（启用增量编译时）：  
> 缓存编译中间状态，加快后续构建速度。

---

### ✅ 执行程序输出

```bash
$ ../target/debug/hello
Hello, world!
```

> 说明你的程序编译成功，并且在运行时按预期输出。

---

### 总结图示（结构一览）

```bash
target/debug/
├── build/          # 构建缓存（宏、构建脚本等）
├── deps/           # 所有依赖包的编译产物
├── examples/       # 示例程序产物
├── hello           # 主可执行程序
├── hello.d         # 可执行程序依赖描述
├── incremental/    # 增量编译缓存
```

---

如需进一步分析二进制大小组成（如哪些 crate 占用了多少空间），可使用工具：

```bash
cargo bloat --release --bin hello
```