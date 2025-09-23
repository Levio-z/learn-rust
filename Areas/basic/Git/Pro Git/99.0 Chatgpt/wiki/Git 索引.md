### Git Index 文件格式详解

Git 的索引（index，也称为缓存）是 Git 存储暂存区状态的核心文件，它用来记录工作区文件与对象库（object database）之间的映射关系。以下内容按 Git 官方文档和源码解析整理。

---

### 1\. 文件头（Header）

索引文件前 12 字节包含：

| 字段 | 长度 | 内容 |
| --- | --- | --- |
| Signature | 4 字节 | `'DIRC'`（dircache 的缩写） |
| Version | 4 字节 | 支持版本：2、3、4 |
| Number of entries | 4 字节 | 32 位整数，索引条目数量 |

**说明**：版本号不同可能会影响索引条目的编码方式，尤其是 V4 引入了路径压缩。

---

### 2\. 索引条目（Index Entry）

索引条目按文件名（`name`）升序排列，相同文件名按 stage 排序。每条索引条目记录文件的元数据和对应的对象 ID。

#### 基本字段（以版本 2 为例）

| 字段          | 长度        | 说明                                                                              |
| ----------- | --------- | ------------------------------------------------------------------------------- |
| ctime       | 32-bit 秒  | 文件元数据最后修改时间                                                                     |
| ctime\_nano | 32-bit 纳秒 | 文件元数据最后修改时间精度                                                                   |
| mtime       | 32-bit 秒  | 文件内容最后修改时间                                                                      |
| mtime\_nano | 32-bit 纳秒 | 文件内容最后修改时间精度                                                                    |
| dev         | 32-bit    | 设备号（stat）                                                                       |
| ino         | 32-bit    | inode 号（stat）                                                                   |
| mode        | 32-bit    | 高 4-bit: 对象类型（1000=普通文件, 1010=符号链接, 1110=gitlink）9-bit UNIX 权限（普通文件仅 0644/0755） |

#### Flags 字段（16-bit）

| Bits   | 说明                      |
| ------ | ----------------------- |
| 1-bit  | assume-valid            |
| 1-bit  | extended flag（v2 必须为 0） |
| 2-bit  | stage（冲突阶段）             |
| 12-bit | 文件名长度（0xFFF 表示超过长度）     |

V3 及以上版本，extended flag=1 时还有 16-bit 扩展字段，用于 skip-worktree、intent-to-add 等。

---

### 3\. 扩展（Extensions）

Git 索引允许可选扩展，用来加速或记录特殊信息：

| 扩展 | Signature | 功能 |
| --- | --- | --- |
| Cache Tree | `TREE` | 缓存树对象，用于加速树对象生成 |
| Resolve Undo | `REUC` | 保存冲突未解决状态，用于恢复冲突 |
| Split Index | `link` | 拆分索引，部分索引条目在共享索引中 |
| Untracked Cache | `UNTR` | 缓存未跟踪文件信息 |
| File System Monitor | `FSMN` | 追踪核心 fsmonitor 钩子通知的变更 |
| End of Index Entry | `EOIE` | 指向索引条目结尾的偏移量，便于快速定位扩展 |
| Index Entry Offset Table | `IEOT` | 支持多线程加载索引条目 |
| Sparse Directory | `sdir` | Sparse checkout 下的目录条目压缩 |

**说明**：

-   扩展由 4 字节签名 + 32-bit 大小 + 数据组成。
    
-   可选扩展（签名首字母 `A..Z`）如果不被理解可以忽略。
    

---

### 4\. 索引条目排序与解析

-   条目按路径名排序，路径名采用 memcmp 比较（无本地化）。
    
-   Stage 不同的条目用于冲突场景，stage 0 为普通条目。
    
-   V4 索引采用前缀压缩：每条条目存储相对于前一条的路径差异（长度 + NUL 终止字符串）。
    

---

### 5\. Cache Tree 扩展细节

-   缓存树扩展记录已有 tree 对象结构，避免重新生成未修改的部分。
    
-   条目包括：
    
    1.  NUL 终止路径
        
    2.  Index entry 数量（ASCII）
        
    3.  Subtree 数量（ASCII）
        
    4.  换行符
        
    5.  Object ID
        
-   如果节点无效，entry\_count = -1，OID 不存在。
    

---

### 6\. 其他重要扩展

-   **Resolve Undo (`REUC`)**：保存 stage 1–3 条目，用于冲突回滚。
    
-   **Split Index (`link`)**：配合共享索引文件，减少重复索引加载。
    
-   **Untracked Cache (`UNTR`)**：存储未跟踪文件的 stat 和哈希信息。
    
-   **File System Monitor (`FSMN`)**：记录文件变更的时间戳或标记。
    
-   **End of Index Entry (`EOIE`)**：快速定位索引条目结尾。
    
-   **Index Entry Offset Table (`IEOT`)**：用于多线程加载索引。
    
-   **Sparse Directory (`sdir`)**：cone 模式下 sparse-checkout 支持目录级压缩。
    

---

### 7\. 总结与方法论

1.  **索引核心概念**：
    
    -   文件元数据 + SHA-1/256 对象 ID + 路径 + flags。
        
    -   支持版本控制、冲突处理、稀疏 checkout 和性能优化。
        
2.  **扩展机制**：
    
    -   扩展可选且版本兼容。
        
    -   EOIE、IEOT 等扩展专注于性能。
        
    -   REUC、UNTR、FSMN 等扩展用于状态管理和缓存。
        
3.  **学习方法论**：
    
    -   理解索引结构对 Git 内部操作的影响，如 `git add`、`git status`。
        
    -   实际解析索引文件（`.git/index`），观察不同版本和扩展数据。
        
    -   通过源码查看 `cache.h`、`index.h`、`read-cache.c` 等解析实现。
        
4.  **重点底层知识**：
    
    -   索引条目字段及 flags 位编码。
        
    -   前缀压缩（V4）与 sparse-checkout 的目录条目压缩。
        
    -   扩展机制（TREE/REUC/link/UNTR/EOIE/IEOT）。
        
    -   索引排序与冲突 stage。
        
5.  **练习建议**：
    
    -   编写工具解析 `.git/index` 文件并打印条目。
        
    -   比较不同版本索引文件（V2/V3/V4）结构差异。
        
    -   添加/删除文件观察索引变化，分析 flags、stage 和扩展数据。
        
    -   尝试解析 sparse-checkout 索引，理解前缀压缩算法。
        

---