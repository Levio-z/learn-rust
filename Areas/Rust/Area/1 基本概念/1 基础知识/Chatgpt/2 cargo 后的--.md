### 为什么 `cargo test -- --show-output` 有两条 `--`

这是因为 **Cargo 本身** 和 **测试二进制** 都能接收命令行参数。  
`cargo test` 执行时会经历两层：

---

#### 1\. 第一层参数：传给 `cargo`

比如：

```bash
cargo test --release
```

这里的 `--release` 是 Cargo 的参数，用来告诉 Cargo **以 release 模式编译**。

---

#### 2\. 第二层参数：传给测试二进制

Cargo 在编译好测试二进制（位于 `target/debug/deps/...`）后，会运行它。  
这个测试二进制是由 **Rust test harness（测试运行器）** 生成的，它也有自己的参数，比如：

-   `--show-output` → 显示所有测试的输出（即使通过也显示 `println!`)。
    
-   `--nocapture` → 不捕获输出，直接打印。
    
-   `--test-threads=N` → 控制并行线程数。
    

---

#### 3\. `--` 的作用：分隔参数归属

Unix 命令行约定：

-   **第一个 `--`**：告诉 Cargo，后面的参数不再是给 Cargo 的，而是要原样转交给测试二进制。
    
-   **第二个 `--` 及之后的内容**：就是测试二进制能识别的参数。
    

所以：

```bash
cargo test -- --show-output
```

解释为：

1.  `cargo test` → 编译测试二进制。
    
2.  第一个 `--` → 参数分隔符，后面的都交给测试二进制。
    
3.  `--show-output` → 测试运行器参数，控制输出行为。
    

---

#### 4\. 举个例子

```bash
cargo test --release -- --nocapture --test-threads=1
```

拆分：

-   `--release` → Cargo 参数（编译模式）。
    
-   第一个 `--` → 分隔符。
    
-   `--nocapture --test-threads=1` → 测试二进制参数（输出 & 并行）。
    

---

### 总结

-   **第一层 `--`**：告诉 Cargo，把后面参数传给测试运行器。
    
-   **第二层及之后的参数**：测试二进制本身的参数，比如 `--show-output`、`--nocapture`。
    
-   这是 Cargo 和测试运行器 **共享命令行参数空间** 的解决方案。
    
