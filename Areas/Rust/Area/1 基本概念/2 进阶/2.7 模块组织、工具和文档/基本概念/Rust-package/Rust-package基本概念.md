
## 1. 核心观点  
### Ⅰ. 概念层
### 定义
_软件包_是一组**源文件**和一个用于**描述该软件包的 `Cargo.toml`** [Rust-cargo-manifest（Cargo.toml）](../../模块系统和封装/cargo/note/note/atomic/Rust-cargo-manifest（Cargo.toml）.md)文件的集合。软件包具有名称和版本，用于**指定软件包之间的依赖关系**。


>package（项目理解为工程、软件包）
- 一个 `Cargo` 提供的 `feature`，可以用来**构建**、**测试**和**分享包**，是一个 **Cargo 的工作单元**，由 `Cargo.toml` 文件定义。Cargo 传递 crate根文件给 rustc 构建library 或 binary.
- **crate组合**
	- 包中可以包含任意数量的binary crate，但最多只能包含一个Library crate库类型的。包中必须至少包含一个crate，无论是Library crates还是binary crate。
- **crate root规则**
	- Cargo 遵循一个约定，即src/main.rs_是与package同名的binary crate的crate root
	- Cargo 知道如果包目录包含src/lib.rs ，则该package包含一个与package同名的library crate，src/lib.rs是它的crate root
- 例子
	- Cargo 实际上是一个包含您用来构建代码的**命令行工具的binary crate**的package
	- Cargo _package_ 还包含该binary crate所依赖的Library crates。
	- 其他项目可以依赖 Cargo _Library crates来使用 Cargo 命令行工具所使用的相同逻辑。


### Ⅱ. 应用层




### Ⅲ. 实现层

### **IV**.原理层


## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  





## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  

