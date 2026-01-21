---
tags:
  - note
---
## 1. 核心观点  
### Ⅰ. 概念层

YAML文件支持
### Ⅱ. 应用层


### Ⅲ. 实现层

### **IV**.原理层


## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### 简介（Introduction）

Markdown 标记语言旨在**易读、易写、易理解**。这一目标它实现得很好，但其高度灵活性既是优势也是缺点：

- 同一内容可以有多种书写风格，导致格式不统一
- 某些语法在不同解析器中表现不一致，存在兼容性问题
    

因此，需要一种工具来**约束风格、提升一致性、避免易错写法**。  
**markdownlint** 正是为此而生：它是一个 VS Code 扩展，内置了一套 Markdown 规则，用于检查并规范 Markdown 文件的写作风格。

该扩展基于 **Node.js 的 markdownlint 库**（灵感来自 Ruby 版本的 markdownlint），底层由 **markdownlint-cli2** 引擎驱动，同时支持：

- VS Code 编辑器内实时检查
    
- 命令行工具（CI / Script）
    
- GitHub Action（markdownlint-cli2-action）
    

---

### 安装（Install）

**方式一：Quick Open**

1. 打开 VS Code
    
2. 按 `Ctrl+P / ⌘P`
    
3. 输入 `ext install markdownlint`
    
4. 点击 Install → Enable
    

**方式二：扩展面板**

1. 按 `Ctrl+Shift+X / ⇧⌘X`
    
2. 搜索 `markdownlint`
    
3. 安装并启用
    

**方式三：命令行**

```bash
code --install-extension DavidAnson.vscode-markdownlint
```

---

### 使用方式（Use）

当你在 VS Code 中编辑 Markdown 文件时：

- 违反规则的行会被**绿色波浪线**标记
    
- 所有问题可在 **Problems 面板**（`Ctrl+Shift+M`）中查看
    
- 所有 markdownlint 报警均以 `MD###` 开头
    

**交互能力：**

- 悬停查看规则说明
    
- `F8 / Shift+F8` 在问题间跳转
    
- 光标放在问题行，点击 💡 或 `Ctrl+.` 查看：
    
    - 规则说明
        
    - 可用的自动修复
        
    - 官方文档链接
        

默认情况下，markdownlint 会检查所有被 VS Code 识别为 Markdown 的文件类型。

---

### 规则体系（Rules）

markdownlint 使用 **MD001–MD060** 的规则编号体系，覆盖以下维度：

- 标题结构（如层级递增、重复标题）
    
- 列表风格与缩进
    
- 空行、空格、行尾
    
- 代码块与围栏风格
    
- 链接、图片、表格规范
    
- 强调与加粗一致性
    
- 文件整体结构要求
    

> 本质上，这是一套 **“Markdown 语义 + 排版风格 + 工程规范” 的组合规则集**

---

### 可自动修复规则（Auto-fix）

以下规则支持 **Quick Fix / Format 自动修复**（节选）：

- 列表缩进与样式（MD004 / MD005 / MD007）
    
- 多余空格与空行（MD009 / MD012）
    
- 标题格式问题（MD018–MD023）
    
- 代码块、表格、链接格式（MD031 / MD034 / MD058）
    
- 文件结尾换行（MD047）
    

这些修复可以：

- 光标处触发（`Ctrl+.`）
    
- 整文档触发（Format Document）
    
- 保存时自动触发（Code Actions）
    

---

### 命令与格式化（Commands）

markdownlint 注册为 **Markdown 格式化器**，支持：

- `Format Document`
    
- `Format Selection`
    

**推荐配置：保存时自动修复**

```json
"[markdown]": {
  "editor.formatOnSave": true,
  "editor.formatOnPaste": true
},
"editor.codeActionsOnSave": {
  "source.fixAll.markdownlint": true
}
```

---

### 工作区检查（Workspace）

- `markdownlint.lintWorkspace`
    
    - 使用 markdownlint-cli2
        
    - 扫描整个工作区
        
    - 输出到 Terminal + Problems 面板
        

可通过 `markdownlint.lintWorkspaceGlobs` 定制扫描范围。

---

### 启用 / 禁用（Disable）

- 使用 `markdownlint.toggleLinting` 临时关闭或开启
    
- 状态在当前工作区有效，新工作区默认重新启用
    

---

### 配置体系（Configure）

**默认行为：**

- 所有规则启用
    
- `MD013（行长限制）` 默认关闭
    

**配置文件形式（按优先级）：**

1. `.markdownlint-cli2.*`
    
2. `.markdownlint.*`
    
3. VS Code 用户 / 工作区设置
    
4. 默认配置
    

**示例：**

```json
{
  "MD003": { "style": "atx_closed" },
  "MD007": { "indent": 4 },
  "no-hard-tabs": false
}
```

支持 `extends` 继承配置，适合多仓库 / 多子项目统一规范。

---

### 高级配置项（Advanced）

- `markdownlint.focusMode`：忽略光标附近问题
    
- `markdownlint.run`：仅在保存时检查
    
- `markdownlint.severityForError / Warning`：自定义严重级别
    
- `markdownlint.customRules`：加载自定义规则（JS / npm 包）
    

---

### 规则抑制（Suppress）

可在 Markdown 文件中局部禁用规则：

```markdown
<!-- markdownlint-disable MD037 -->
文本内容
<!-- markdownlint-enable MD037 -->
```

---

### 代码片段（Snippets）

内置常用 snippet，如：

- `markdownlint-disable`
    
- `markdownlint-disable-line`
    
- `markdownlint-disable-file`
    
- `markdownlint-configure-file`
    

---

### 安全性（Security）

由于支持执行 JavaScript（自定义规则、插件、配置文件），markdownlint **遵循 VS Code Workspace Trust**：

- 不可信工作区默认禁止 JS 执行
    

---

## 总结

markdownlint 本质上是一个 **Markdown 的静态规范检查器**，目标是：

- 提升文档一致性
    
- 降低 Markdown 在团队协作中的风格摩擦
    
- 将“写作规范”工程化、自动化
    

它在 **技术文档、开源项目、知识库、博客仓库、CI 流水线** 中具有极高价值。

---

## 学习方法论

1. **先启用默认规则**，观察真实问题分布
    
2. **只关闭你“明确理解并不需要”的规则**
    
3. 将 `.markdownlint.json` 纳入仓库，形成团队规范
    
4. 配合 `formatOnSave + fixAll`，做到“无感规范化”
    
5. 在 CI 中接入 markdownlint-cli2，防止规范回退
    

---

## 练习建议

1. 为一个已有 README 启用 markdownlint，分析触发最多的规则
    
2. 手写一份 `.markdownlint.json`，解释每个被关闭规则的理由
    
3. 编写一个自定义规则（如：强制某些标题必须存在）
    
4. 在 GitHub Actions 中加入 markdownlint 检查步骤
    

---

## 高价值底层知识（重点关注）

- Markdown AST 与解析器差异
    
- lint 工具的规则模型设计
    
- VS Code Diagnostic / CodeAction 机制
    
- CLI 与 Editor 共用引擎的工程化思想
    
- “文档即代码（Docs as Code）” 的质量控制体系#### ```


## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
- [x] 深入阅读 xxx
- [x] 验证这个观点的边界条件
