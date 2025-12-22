---
tags:
  - permanent
---
## 1. 核心观点  

`cargo install` 是 Rust 的 **包安装命令**，用于从 **crates.io** 或本地源码安装 **可执行二进制工具**，而不是库（library）。它与 `cargo build` 的主要区别在于安装目标和用途。

## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  

### 🔹 `cargo install` 作用

`cargo install` 是 Rust 的 **包安装命令**，用于从 **crates.io** 或本地源码安装 **可执行二进制工具**，而不是库（library）。它与 `cargo build` 的主要区别在于安装目标和用途。

---

### 🔹 功能与用途

1. **安装可执行文件**
    - 将 Rust crate 编译为可执行程序，并放到 `~/.cargo/bin`（Linux/macOS）或 `%USERPROFILE%\.cargo\bin`（Windows）
    - 安装后，可以在命令行中全局使用该工具
        
2. **常用场景**
    
    - 安装命令行工具，例如：
        ```bash
        cargo install ripgrep
        cargo install exa
        cargo install cargo-edit
        ```
        
    - 开发自己或别人写的 CLI 工具
        
3. **可指定版本**
    
    - 指定 crate 版本安装：
        
        ```bash
        cargo install bat --version 0.23.0
        ```
        
4. **支持路径安装**
    
    - 可以安装本地 crate 的可执行文件：
        ```bash
        cargo install --path ./my_cli_tool
        ```
        
5. **覆盖安装**
    
    - 默认不会覆盖已安装的同名工具，可以加 `--force` 强制覆盖：
        ```bash
        cargo install cargo-edit --force
        ```
        
6. **额外选项**
    
    - `--locked`：使用 Cargo.lock 文件中的版本，保证可复现构建
        
    - `--root <dir>`：指定安装路径（默认 `~/.cargo`）
        

---

### 🔹 总结

- **核心作用**：快速获取 Rust crate 的 **全局可执行工具**
    
- **区别于 `cargo build` / `cargo run`**：
    
    - `cargo build`：编译项目（本地开发用）
    - `cargo run`：编译并运行项目
    - `cargo install`：编译并 **安装可执行文件到系统路径**
        

---

### 🔹 学习方法与实践

1. 尝试安装几个常用工具：`ripgrep`, `exa`, `cargo-edit`
    
2. 查看 `~/.cargo/bin` 是否生成了可执行文件
    
3. 使用 `cargo install --path ./local_tool` 测试本地开发工具安装
    
4. 熟悉常用参数：`--version`, `--force`, `--locked`
    

重点关注 **全局可执行安装机制** 和 **与 cargo build/run 的区别**，是 Rust 工具链管理的底层知识。

---

如果你需要，我可以画一个 **cargo install 的执行流程图**，让你更直观理解编译、安装到 `~/.cargo/bin` 的过程。

你想让我画吗？

## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
- [ ] 深入阅读 xxx  
- [ ] 验证这个观点的边界条件  
