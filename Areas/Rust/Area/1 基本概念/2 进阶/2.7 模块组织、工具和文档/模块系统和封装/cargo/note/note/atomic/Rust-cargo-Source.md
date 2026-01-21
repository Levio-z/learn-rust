---
tags:
  - note
---
## 1. 核心观点  
### Ⅰ. 概念层
| **来源类型**           | **核心关注点** | **典型用户**      |
| ------------------ | --------- | ------------- |
| **Registry**       | 标准化、版本化   | 所有 Rust 开发者   |
| **Local Registry** | 离线、压缩包分发  | 运维、OS 打包者     |
| **Directory**      | 本地备份、审计   | 高安全要求项目       |
| **Path**           | 快速迭代、本地耦合 | 正在编写多模块项目的开发者 |
| **Git**            | 最新代码、简单共享 | 抢鲜测试者、小团队     |
是指包含可作为[_软件包_](https://doc.rust-lang.org/cargo/appendix/glossary.html#package)依赖项的 [_crate_](https://doc.rust-lang.org/cargo/appendix/glossary.html#crate) 的提供程序。源有几种类型：

- **Registry source** — See [registry](https://doc.rust-lang.org/cargo/appendix/glossary.html#registry).  
    **注册表来源** — 请参阅[注册表](https://doc.rust-lang.org/cargo/appendix/glossary.html#registry) 。
    
- **Local registry source** — A set of crates stored as compressed files on the filesystem. See [Local Registry Sources](https://doc.rust-lang.org/cargo/reference/source-replacement.html#local-registry-sources).  
    **本地注册表源** — 一组以压缩文件形式存储在文件系统中的 crate。参见[本地注册表源](https://doc.rust-lang.org/cargo/reference/source-replacement.html#local-registry-sources) 。
- **Directory source** — A set of crates stored as uncompressed files on the filesystem. See [Directory Sources](https://doc.rust-lang.org/cargo/reference/source-replacement.html#directory-sources).  
    **目录源** — 一组以未压缩文件形式存储在文件系统中的 crate。参见[目录源](https://doc.rust-lang.org/cargo/reference/source-replacement.html#directory-sources) 。
- **Path source** — An individual package located on the filesystem (such as a [path dependency](https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html#specifying-path-dependencies)) or a set of multiple packages (such as [path overrides](https://doc.rust-lang.org/cargo/reference/overriding-dependencies.html#paths-overrides)).  
    **路径源** — 文件系统上的单个软件包（例如 [路径依赖](https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html#specifying-path-dependencies) ）或一组多个包（例如[路径覆盖](https://doc.rust-lang.org/cargo/reference/overriding-dependencies.html#paths-overrides) ）。
- **Git source** — Packages located in a git repository (such as a [git dependency](https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html#specifying-dependencies-from-git-repositories) or [git source](https://doc.rust-lang.org/cargo/reference/source-replacement.html)).  
    **Git 源** — 位于 git 存储库中的软件包（例如 [git 依赖项](https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html#specifying-dependencies-from-git-repositories)或 [git 源](https://doc.rust-lang.org/cargo/reference/source-replacement.html) ）。

### Ⅱ. 应用层




### Ⅲ. 实现层

### **IV**.原理层


## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### 1. Registry Source (注册表源)

这是**最常用**的场景。
- **场景**：当你使用公共库（如 `serde`, `tokio`）时。

- **用法**：默认指向 [crates.io](https://crates.io)。你只需在 `Cargo.toml` 中写版本号。

- **企业应用**：大型公司会搭建**私有注册表**（Private Registry），用于存放内部不公开的组件库，同时保证下载速度和安全性。
### 2. Local Registry Source (本地注册表源)

- **场景**：**离线构建**或**分发**。
- **具体用例**：
    - 在没有网络连接的 CI/CD 环境中构建。
    - 作为 Linux 发行版（如 Debian 或 Fedora）的一部分，将所有依赖打包在一个压缩包中进行离线安装。
    - 它比单纯的目录源更正式，因为它包含索引文件，可以像 crates.io 一样被检索。
### 3. Directory Source (目录源)

- **场景**：**Vendoring（供应商模式）**。
    
- **具体用例**：
    
    - 使用 `cargo vendor` 命令将所有远程依赖下载到项目的 `vendor/` 目录下。
        
    - 这在对代码审计要求极高的项目中很常见，确保所有第三方代码都存在于本仓库中，不依赖任何外部服务器。

### 4. Path Source (路径源)

这是**本地多项目开发**的标准方式。

- **场景**：**工作区（Workspaces）**或**本地调试**。
    
- **具体用例**：
    
    - **模块化开发**：你正在开发一个主应用 `app`，同时在旁边写一个库 `my_utils`。在 `app` 的配置里，你会写 `my_utils = { path = "../my_utils" }`。
        
    - **临时修复**：你发现某个第三方库有 Bug，把它克隆到本地修改，然后通过 `[patch]` 临时指向这个本地路径进行测试。

### 5. Git Source (Git 源)

- **场景**：**抢鲜体验**或**内部私有化**。
    
- **具体用例**：
    
    - **使用未发布版本**：某个库修复了一个 Bug，但还没发布到 crates.io，你可以直接指向该库的 GitHub 仓库地址或某个具体的 Commit。
        
    - **私有 Git 仓库**：小团队内部没有搭建复杂的 Registry，直接通过 `git = "ssh://..."` 引用内部服务器上的代码仓库。



## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  

