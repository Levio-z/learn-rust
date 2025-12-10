### 🔹 `pip` 与 `pipx` 的区别

#### 1. **pip**

- **用途**：Python 官方包管理器，用于安装 Python 库和工具。
    
- **安装位置**：
    
    - 默认安装到当前 Python 环境（系统 Python 或虚拟环境）。
        
    - 系统环境安装：`sudo pip install package`（可能覆盖系统库，引起冲突）。
        
    - 虚拟环境安装：安全隔离，使用 `python -m venv venv && source venv/bin/activate`。
        
- **适合场景**：
    
    - 安装 Python 库（比如 `requests`、`numpy`）供程序依赖使用。
        
    - 项目内依赖管理，通常配合 `requirements.txt`。
        

---

#### 2. **pipx**

- **用途**：专门用于安装 **Python 命令行应用程序（CLI）**。
    
- **核心特点**：
    
    - 为每个应用创建独立虚拟环境。
        
    - 全局可执行命令自动链接到 PATH，无需手动激活虚拟环境。
        
    - 不污染系统 Python 环境，避免依赖冲突。
        
- **安装方式**：
    
    ```bash
    pipx install package_name
    ```
    
    - 比如 `pipx install pre-commit`，可以直接在终端运行 `pre-commit` 命令。
        
- **适合场景**：
    
    - 安装 CLI 工具，如 `black`, `pre-commit`, `httpie`。
        
    - 想全局可用，但不破坏系统 Python 环境时。
        

---

#### 3. **对比总结表**

|特性|pip|pipx|
|---|---|---|
|目标|库、工具|CLI 工具|
|安装位置|当前 Python 环境|独立虚拟环境 + PATH|
|系统污染|可能覆盖系统库|不污染系统环境|
|全局可用|需配置环境或虚拟环境激活|自动全局可用|
|推荐使用场景|项目依赖管理|命令行工具全局使用|

---

✅ **总结**

- **pip** → 主要针对库和项目依赖
    
- **pipx** → 主要针对 CLI 工具，安全全局使用
    
- 如果你只想快速安装并运行一个 Python CLI 工具，`pipx` 更安全方便。
    

---

如果你想，我可以帮你画一个小图，把 `pip` 和 `pipx` 的安装路径和命令可执行性对比可视化，让概念更直观。你希望我画吗？