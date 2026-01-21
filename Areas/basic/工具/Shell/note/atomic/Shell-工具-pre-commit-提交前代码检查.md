---
tags:
  - note
---
## 1. 核心观点  
### Ⅰ. 概念层
- 基本概念
    - 定义: pre-commit是一个代码检查工具，可以在提交代码前进行代码检查。
    - 作用: 通过自动化检查避免将低级错误提交到代码库，提高代码质量。
    - 配置文件: 使用.pre-commit-config.yaml文件进行配置。

### Ⅱ. 应用层
- [使用流程](#使用流程)

### Ⅲ. 实现层
- [配置说明](Shell-工具-pre-commit-提交前代码检查.md#配置说明)

### **IV**.原理层


## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### 安装方法
- pip安装:
	- 命令: pip install pre-commit
	- 特点: 全局安装
- pipx安装:
	- 命令: pipx install pre-commit
	- 特点: 隔离环境安装，避免全局污染
- 其他方式:
	- Mac用户可使用homebrew安装
	- Conda用户可使用conda install -c conda-forge pre-commit
### 配置说明
- 基本结构:
	- Rust相关钩子:
		- cargo-fmt: 代码格式化
		- cargo-deny: 依赖安全检查
		- typos: 拼写检查
		- cargo-check: 基础编译检查
		- cargo-clippy: 代码lint检查
		- cargo-test: 单元测试
### 使用流程
- 初始化:
	- **运行pre-commit install安装git钩子**
	- 钩子会自动安装在.git/hooks/pre-commit
- 首次检查:
	- 建议运行**pre-commit run --all-files**对所有文件进行检查
- 后续使用:
	- **每次git commit时自动运行检查**
	- 只检查被修改的文件

- 注意事项
    - 版本控制:
        - **建议将.pre-commit-config.yaml加入版本控制**
        - 团队共享相同配置
    - 性能影响:
        - 复杂的检查可能增加提交时间
        - 可通过exclude配置排除不需要检查的文件
    - 错误处理:
        - fail_fast选项控制是否在第一个错误时停止
        - 建议开发阶段设为false，查看所有问题
### 高级功能
- 多语言支持:
	- 支持Python、Perl、R、Ruby等多种语言
	- 每种语言有特定的安装和执行方式
- 环境隔离:
	- 自动创建隔离环境运行钩子
	- 不会污染系统环境
- 缓存机制:
	- 安装的钩子会被缓存
	- 后续运行会重用缓存提高速度
## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
	- [shell-工具-pre-commit- 配置文件解析](../misc/shell-工具-pre-commit-%20配置文件解析.md)
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
- [x] 深入阅读 xxx
- [x] 验证这个观点的边界条件
