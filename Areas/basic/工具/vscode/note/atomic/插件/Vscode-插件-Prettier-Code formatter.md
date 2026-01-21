---
tags:
  - note
---
## 1. 核心观点  
### Ⅰ. 概念层

Prettier 是一个**有明确格式规范（opinionated）**的代码格式化工具。它通过解析你的代码并依据自身规则重新打印，从而实现**统一风格**的格式化，并在必要时根据最大行宽自动换行。

支持语言包括：  

**JavaScript · TypeScript · Flow · JSX · JSON · CSS · SCSS · Less · HTML · Vue · Angular · Handlebars · Ember · Glimmer · GraphQL · Markdown · YAML**


### Ⅱ. 应用层




### Ⅲ. 实现层

### **IV**.原理层


## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  


### ⚙️ 设置默认格式化工具

为了确保 Prettier 被优先使用，请在 `settings.json` 中设置：

```json
{
  "editor.defaultFormatter": "esbenp.prettier-vscode",
  "[javascript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  }
}
```

如果想对某语言禁用 Prettier：

```json
{
  "editor.defaultFormatter": "esbenp.prettier-vscode",
  "[javascript]": {
    "editor.defaultFormatter": "<other-formatter>"
  }
}
```

也可以反过来，仅对特定语言启用 Prettier：

```json
{
  "editor.defaultFormatter": "<other-formatter>",
  "[javascript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  }
}
```

---

### 💾 保存时自动格式化

```json
"editor.formatOnSave": false,
"[javascript]": {
  "editor.formatOnSave": true
}
```

---

### 📦 Prettier 解析优先级

扩展会优先使用**项目本地安装的 Prettier**（推荐方式）。  
若启用 `"prettier.resolveGlobalModules": true`，则也会尝试解析全局模块。  
若都找不到，则使用**扩展自带版本**。

安装推荐命令：

```bash
npm install prettier -D --save-exact
```

---

### 🧩 Prettier 插件支持

若项目中在 `package.json` 注册了 Prettier 插件（如 HTML、Rust、SQL 格式化插件），  
扩展将自动检测并启用这些语言的格式化支持。

---

### ⚙️ 配置方式

Prettier 支持多种配置来源（按优先级排列）：

1. `.prettierrc` 或 `.prettierrc.json/.yaml/.js`
    
2. `.editorconfig`
    
3. VS Code 设置（仅在项目未配置时生效）
    

推荐做法：  
**在项目根目录下添加 `.prettierrc` 文件，确保所有环境下格式一致。**

示例：

```json
{
  "printWidth": 100,
  "singleQuote": true,
  "semi": false,
  "trailingComma": "es5"
}
```

如果不想每个项目都配置，可设置全局默认：

```json
"prettier.configPath": "/path/to/global/.prettierrc"
```

---

### 🖱️ 使用方式

#### 命令面板

`Cmd/Ctrl + Shift + P → Format Document`  
或选中文本后执行 `Format Selection`。

#### 快捷键

- Windows/Linux: `Shift + Alt + F`
    
- macOS: `Option + Shift + F`
    

#### 强制格式化

如果文件在 `.prettierignore` 或 `node_modules` 内，可用：

```
Format Document (Forced)
```

---

### 🧹 Linter 集成建议

最佳实践：

- **让 Prettier 处理格式**
    
- **让 ESLint/TSLint 仅处理语义与逻辑**
    

在 Prettier 官方文档中有针对 ESLint 的详细配置指导。

---

### 🏠 工作区信任机制（Workspace Trust）

在未信任的工作区中：

- 仅使用内置版本的 Prettier；
    
- 不加载本地或全局模块；
    
- 插件与部分配置被禁用。
    

---

### 🔧 可配置项（部分）

|设置项|默认值|功能说明|
|---|---|---|
|`prettier.enable`|`true`|启用/禁用 Prettier|
|`prettier.requireConfig`|`false`|是否必须存在配置文件才能格式化|
|`prettier.ignorePath`|`.prettierignore`|忽略文件路径|
|`prettier.configPath`|—|自定义配置文件路径|
|`prettier.prettierPath`|—|指定 Prettier 模块路径|
|`prettier.resolveGlobalModules`|`false`|允许使用全局 Prettier|
|`prettier.withNodeModules`|`false`|是否格式化 node_modules 内文件|
|`prettier.useEditorConfig`|`true`|读取 `.editorconfig` 配置|
|`prettier.documentSelectors`|—|绑定文件类型（如 `"**/*.abc"`）|

---

### ⚠️ 常见错误信息

|错误提示|解决方式|
|---|---|
|`Failed to load module`|运行 `npm install` 安装依赖|
|`Outdated prettier version`|升级 Prettier 到最新版本|
|`Untrusted workspace`|信任工作区以启用本地模块与插件|

---

### 🧠 总结与学习方法

**核心价值：**  
Prettier 的目标是统一代码风格，减少风格争论，让开发者专注于逻辑本身。

**学习建议：**

1. 实践配置 `.prettierrc` 并理解每个选项的影响；
    
2. 理解 Prettier 与 ESLint 的分工；
    
3. 尝试在多人项目中共享配置；
    
4. 了解插件系统以支持更多语言格式化。
    

**练习方向：**

- 编写同一段代码，比较开启/关闭 `semi`、`singleQuote`、`trailingComma` 的差异；
    
- 手动创建 `.prettierignore` 文件并测试生效；
    
- 实验 VS Code 中 `formatOnSave` 和 `defaultFormatter` 的优先级行为。
    

重点掌握的底层知识包括：

- AST（抽象语法树）在格式化工具中的作用；
    
- VS Code 的扩展机制与语言服务（LSP）；
    
- Prettier 插件注册与文件类型映射机制。

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
