ls -l ~/.cargo/bin
### 🔹 命令解析

- `ls` ：列出目录内容（list）
    
- `-l` ：使用 **长列表格式**（long format），显示详细信息，包括：
    
    - 文件类型（普通文件、目录、符号链接等）
        
    - 权限（rwx）
        
    - 拥有者（user）
        
    - 用户组（group）
        
    - 文件大小（bytes）
        
    - 最后修改时间
        
    - 文件名
        
- `~/.cargo/bin` ：指定目录路径
    
    - `~` 表示当前用户的 **home 目录**
        
    - `.cargo/bin` 是 Rust 工具链默认安装的 **可执行文件目录**
        
        - 例如 `cargo`、`rustc`、`rustup` 等会在这里创建符号链接或可执行文件