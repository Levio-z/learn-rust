---
tags:
  - note
---
## 1. 核心观点  

`deltification` 是 Git 存储优化的核心机制之一，用于在对象数据库（object database）中通过**差异压缩（delta compression）**减少冗余数据。
## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### 基本概念
Git 中的每个对象（blob、tree、commit、tag）都可以单独存储为完整内容（称为 **base object**），也可以存储为“与另一个对象的差异”（称为 **delta object**）。

Git 的 `packfile` 就是由大量 base 对象和 delta 对象组成的压缩包，其中：

- **base object**：存放原始完整内容。
- **delta object**：存放相对于某个 base 的差异信息。
### 多层 deltification（递归 delta）

Git 允许一个 delta 的基础对象（base）本身也是 delta，从而形成**多层 deltification**。  
这种链式结构会进一步减小整体存储体积，但在解包时需要多次递归应用 delta，直到还原出最初的完整对象。
例：
```rust
eeac5890e8e6b7500c12ebd51af4f6c6be6f9ccc  → delta
  ↳ 基础对象：9e08d4ffcc8beb10dc6eaaa93cdbdb00cfc95be9  → delta
       ↳ 基础对象：be5d471b02b9ee56538c8dc04253ec7206a43971  → base
```
因此，`eeac58...` 是一个两层 delta（depth = 2）。  
Git 在 packfile 中会用一个字段记录 delta 的“深度”，用于快速判断递归层数，防止链过长导致解压低效。
### 底层结构与索引意义
在 `.pack` 文件中，每个对象记录以下信息：
1. 类型（base 或 delta）
2. 对象大小
3. 若为 delta，则记录：
    - 基础对象偏移量或哈希
    - delta 指令序列（基于基础对象生成目标对象）
4. 若为 deltified，则额外有一个字段记录 **delta depth**
这让 Git 的对象存储形成了一个**有向无环图（DAG）**，叶节点为完整对象，其他节点通过 delta 链接到它。  
因此，“不要被十六进制名称所迷惑”是指这些哈希只是节点标识符，本质上这是一种递归的结构化压缩。
### 两种类型
有两种类型的 deltified 对象。如果类型是 `OBJ_REF_DELTA`（引用 delta 对象），则对象元信息（我们之前已经解析过）后面跟着基本对象的 20 字节名称。请记住，delta 类似于 diffs，因此我们需要知道我们将要应用的 diff 的参考点是什么。在此名称之后，您将找到 zlib 压缩的 delta 数据，我们稍后将处理这些数据。

然而，即使这样也很浪费，因此 Git 添加了 `OBJ_OFS_DELTA`（偏移量增量对象）。对于这些，没有 20 字节的基本对象名称，而是另一个可变长度的整数。同样，MSB 告诉我们整数的最后一个字节在哪里。这个整数代表一个负偏移量——它告诉我们在当前包文件中向后扫描多少字节才能找到基本对象。负偏移量后面是 zlib 压缩的增量数据，就像 `OBJ_REF_DELTA` 一样。
### 两种增量对象类型


#### (2) **OBJ_OFS_DELTA（偏移型增量）**
[Git-负偏移量的定义](Git-负偏移量的定义.md)
- **格式结构：**
```rust
[type/size header] 
[base offset (variable-length integer)] 
[zlib-compressed delta data]
```
- **说明：**
    - 没有哈希，而是一个 **可变长度整数（variable-length integer）**。
    - 表示 **当前对象相对于基对象的负偏移量**。
    - Git 在解析时需“向后扫描” packfile 若干字节找到基对象。
> 例：若该整数的解码值为 `0x13F`，表示基对象在当前对象开始前 `0x13F` 字节处。

### 应用增量

我们快完成了！我们已经阅读了整个索引文件和包文件，我们有一堆基本对象和增量对象。我们唯一需要做的就是将增量应用于基本对象以重建缺失的对象。
## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
