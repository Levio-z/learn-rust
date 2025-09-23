`PathBuf` 是 Rust 标准库中 `std::path` 模块提供的可增长、可修改的路径类型，是 `Path` 的可变拥有者版本。熟练掌握核心 20% 的方法可以覆盖绝大多数文件系统操作需求。

---

### 1\. 定义与创建

```rust
use std::path::PathBuf;

// 从字符串创建
let mut path = PathBuf::from("/home/user");

// 空路径
let mut empty = PathBuf::new();
```

-   **`PathBuf::from`**：常用方法，将 `&str` 或 `String` 转换为 `PathBuf`。
-   **`PathBuf::new`**：创建空路径，可动态追加组件。
---

### 2\. 路径修改

`PathBuf` 可变，常用修改方法：
```rust
let mut path = PathBuf::from("/home/user");
// 追加组件
path.push("docs"); // /home/user/docs
// 移除最后一个组件
path.pop(); // /home/user
// 替换文件名
path.set_file_name("file.txt"); // /home/user/file.txt
// 替换扩展名
path.set_extension("md"); // /home/user/file.md
```

-   **`push`**：追加目录或文件名。
    
-   **`pop`**：移除最后一级路径。
    
-   **`set_file_name`**：修改最后一级组件（文件或目录名）。
    
-   **`set_extension`**：修改文件扩展名。
    

---

### 3\. 路径信息查询

```rust
use std::path::Path;

let path = Path::new("/home/user/file.txt");

// 判断路径类型
println!("Has file name? {}", path.file_name().is_some());
println!("Extension? {:?}", path.extension());
println!("Parent? {:?}", path.parent());
println!("Is absolute? {}", path.is_absolute());
println!("Is relative? {}", path.is_relative());
println!("Path is: {}", path.display());
```
-   **`file_name()`**：获取最后一级组件。
-   **`extension()`**：获取文件扩展名。
-   **`parent()`**：获取上级目录。
-   **`is_absolute()` / `is_relative()`**：判断路径类型。
-    **`display`**： 是 `std::path::Path` 提供的一个便捷方法，用于生成一个 **可打印的、平台无关的路径表示**，主要用于调试、日志和输出。

---

### 4\. 转换与拼接

```rust
let path = PathBuf::from("/home/user/file.txt");
// 转换为 &Path
let path_ref: &Path = path.as_path();
// 转换为字符串（可选）
let path_str = path.to_str(); // Option<&str>
// 克隆与组合
let mut new_path = path.clone();
new_path.push("subdir"); // /home/user/file.txt/subdir
```
-   **`as_path()`**：获取不可变 `&Path`。
-   **`to_str()`**：尝试转成 UTF-8 字符串，返回 `Option<&str>`。
-   **`clone()`**：生成新路径用于组合。
---

### 5\. 常用组合用法示例

```rust
use std::path::PathBuf;

let mut path = PathBuf::from("/home/user");
path.push("docs");
path.push("file.txt");

if let Some(ext) = path.extension() {
    println!("File extension: {:?}", ext);
}

path.set_extension("md");
println!("New path: {:?}", path);
```

输出：
```pgsql
File extension: "txt"
New path: "/home/user/docs/file.md"
```

---

### 6\. 总结与学习方法论

-   **核心用途**：
    
    1.  动态构造和修改文件路径。
        
    2.  获取文件名、扩展名、父目录等信息。
        
    3.  与文件系统操作 (`std::fs` / `tokio::fs`) 配合使用。
        
-   **底层价值**：
    
    -   跨平台路径处理（Windows vs Unix）。
        
    -   避免手动字符串拼接，防止路径分隔符错误。
        
-   **学习方法**：
    
    -   实践 `push`、`pop`、`set_file_name`、`set_extension`。
        
    -   使用 `as_path()` 传递给文件 I/O API。
        
    -   测试绝对路径、相对路径和符号链接组合的行为。
        
-   **重点掌握**：
    
    -   `push` / `pop`（路径修改）
        
    -   `file_name` / `extension` / `parent`（信息获取）
        
    -   `as_path` / `to_str`（与 API 转换）
        
-   **练习题**：
    
    1.  构建一个动态路径 `/home/user/logs/YYYY/MM/DD.log`。
        
    2.  将相对路径转换为绝对路径并打印父目录。
        
    3.  遍历目录列表，过滤出特定扩展名的文件路径。
        

---

如果你需要，我可以帮你整理一张 **PathBuf 常用方法速查表**，只包含最核心的 20% 方法和示例，非常适合快速查阅。你想让我做吗？