---
tags:
  - note
---

## 1. 核心观点  
使用 `os_pipe` 跨平台库（推荐）

## 2. 展开说明  

在 Rust 中创建管道（Pipe）的方式取决于您需要的管道类型：**匿名管道** 或 **命名管道（FIFO）**。

### 1. 匿名管道（Anonymous Pipe）

匿名管道是**单向**的，通常用于**相关进程**（如父进程和子进程）之间的通信，并且在进程退出时自动销毁。

在 Rust 中，可以使用标准库 `std::io::pipe` 函数来创建匿名管道。这个功能目前在稳定版 Rust 中还**没有直接暴露在 `std` 中**，但可以通过**第三方跨平台库 `os_pipe`** 来方便地实现，或者在 Unix-like 系统上使用特定平台的 API。

#### 使用 `os_pipe` 跨平台库（推荐）

这是最推荐的跨平台方式。您需要在 `Cargo.toml` 中添加依赖：

Ini, TOML

```
[dependencies]
os_pipe = "0.9"
```

**示例代码：**

Rust

```
use os_pipe::pipe;
use std::io::{self, Read, Write};

fn main() -> io::Result<()> {
    // 1. 创建匿名管道，返回 (读取端, 写入端)
    let (mut read_pipe, mut write_pipe) = pipe()?;

    let data_to_send = "Hello from the writer side!";

    // 2. 写入数据到管道
    println!("写入端正在发送数据: '{}'", data_to_send);
    write_pipe.write_all(data_to_send.as_bytes())?;
    
    // 关闭写入端非常重要，否则读取端可能永远不会收到 EOF
    // 如果不关闭，read_to_string可能会一直阻塞
    drop(write_pipe); 

    // 3. 从管道读取数据
    let mut buffer = String::new();
    read_pipe.read_to_string(&mut buffer)?;

    println!("读取端收到的数据: '{}'", buffer);

    Ok(())
}
```

---

### 2. 命名管道（Named Pipe / FIFO）

命名管道（在 Unix-like 系统中也叫 **FIFO**）是一个特殊类型的文件，存在于文件系统中。它允许**不相关进程**之间通过文件路径进行通信，并且其生命周期**独立于创建它的进程**。

创建命名管道需要使用操作系统底层的特定函数。在 Rust 中，通常需要依赖**第三方库**或使用 **C 语言的 `libc` 接口**。

#### 使用 `nix` 库创建 FIFO (Unix-like 系统)

在 Unix-like 系统（Linux, macOS, BSD）上，使用 `mkfifo` 系统调用来创建 FIFO。您需要在 `Cargo.toml` 中添加依赖：

Ini, TOML

```
[dependencies]
nix = { version = "0.27", features = ["fs"] }
```

**示例代码：**

Rust

```
use nix::sys::stat::{Mode, mkfifo};
use std::fs::File;
use std::io::{self, Write, Read};
use std::path::Path;

fn main() -> io::Result<()> {
    // 命名管道的文件路径
    let fifo_path = Path::new("/tmp/my_rust_fifo");

    // 1. 创建命名管道 (mkfifo)
    // Mode::S_IFIFO | Mode::S_IRWXU 表示创建 FIFO 类型文件，并设置文件权限
    match mkfifo(fifo_path, Mode::S_IFIFO | Mode::S_IRWXU) {
        Ok(_) => println!("成功创建命名管道: {}", fifo_path.display()),
        Err(e) => {
            // 如果文件已存在，则忽略错误
            if e.desc() != "File exists" {
                eprintln!("创建 FIFO 失败: {}", e);
                return Err(io::Error::new(io::ErrorKind::Other, e));
            }
        }
    }

    // --- 写入端示例 ---
    let data_to_send = "Hello via Named Pipe!";
    println!("尝试以写入模式打开 FIFO...");
    
    // 打开 FIFO 时会阻塞，直到有另一个进程以读取模式打开它
    let mut writer = File::create(fifo_path)?;
    
    println!("写入端已连接。写入数据...");
    writer.write_all(data_to_send.as_bytes())?;
    writer.flush()?;
    drop(writer); // 关闭写入端

    // --- 读取端示例 (通常在另一个独立进程中运行) ---
    // 为了在这个程序中演示，我们重新打开文件：
    println!("尝试以读取模式打开 FIFO...");
    let mut reader = File::open(fifo_path)?;
    
    let mut buffer = String::new();
    reader.read_to_string(&mut buffer)?;
    
    println!("读取端收到的数据: '{}'", buffer);

    // 3. 可选：清理（删除 FIFO 文件）
    std::fs::remove_file(fifo_path)?;
    println!("已删除命名管道文件。");

    Ok(())
}
```

#### Windows 命名管道

在 Windows 上，命名管道的行为和 API 与 Unix FIFO 有很大不同。您通常需要使用专门针对 Windows API 的 Rust 库，如 `windows-named-pipe` 或 `miow`，它们封装了 `CreateNamedPipe` 等 Windows API。由于 Windows 命名管道更复杂，并且通常涉及服务器/客户端模型，这里不提供简化的跨平台示例


## 3. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 4. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  

