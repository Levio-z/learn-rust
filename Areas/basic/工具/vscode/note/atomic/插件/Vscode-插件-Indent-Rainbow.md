---
tags:
  - note
---
## 1. 核心观点  
### Ⅰ. 概念层
**一个让缩进更易读的简单扩展**它会在每一级缩进处循环使用四种不同的颜色，让代码层次一目了然。  


### Ⅱ. 应用层




### Ⅲ. 实现层

### **IV**.原理层


## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### Indent-Rainbow



---

#### 🧩 插件简介

Indent-Rainbow 是一款 VS Code 插件，用于给代码缩进区域着色。

这在编写 **Python、Nim、YAML** 等基于缩进语义的语言时尤其有帮助，也能在其他语言中帮助你发现缩进错误。

💡 自 v8.0.0 起，该插件已支持 **vscode-web（github.dev）**。

#### 🌈 插件效果

插件会根据当前编辑器的 **tab 宽度** 自动调整缩进显示，并能正确处理混合的空格与制表符（虽然不推荐混用）。  
此外，插件会高亮显示那些缩进不是 tabSize 倍数的行，帮助你快速定位缩进错误。

---

#### ⚙️ 配置选项

你可以直接使用默认配置，也可以自定义以下设置：

```jsonc
// 启用的语言列表（空数组表示全部语言启用）
"indentRainbow.includedLanguages": ["python", "nim", "yaml"],

// 禁用的语言列表（空数组表示不禁用任何语言）
"indentRainbow.excludedLanguages": ["plaintext"],

// 编辑器更新延迟（毫秒）。数值越小刷新越快，但资源占用更高。
"indentRainbow.updateDelay": 100,
```

> ⚠️ 注意：同时设置 `includedLanguages` 和 `excludedLanguages` 没有意义，请只使用其中之一。

---

#### 🎨 自定义颜色

你可以定义自己的配色方案。例如针对深色背景：

```jsonc
"indentRainbow.colors": [
  "rgba(255,255,64,0.07)",
  "rgba(127,255,127,0.07)",
  "rgba(255,127,255,0.07)",
  "rgba(79,236,236,0.07)"
],
// 缩进不是 tabSize 倍数时的错误颜色
"indentRainbow.errorColor": "rgba(128,32,32,0.6)",
// 混用空格和 Tab 时的颜色（设为空字符串可禁用）
"indentRainbow.tabmixColor": "rgba(128,32,96,0.6)"
```

---

#### 💡 浅色模式（v8.3.0 新增）

在浅色主题中，你可以启用 **线条模式（light mode）**，使用线条代替背景色：

```jsonc
"indentRainbow.indicatorStyle": "light",
"indentRainbow.lightIndicatorStyleLineWidth": 1,
"indentRainbow.colors": [
  "rgba(255,255,64,0.3)",
  "rgba(127,255,127,0.3)",
  "rgba(255,127,255,0.3)",
  "rgba(79,236,236,0.3)"
]
```

> 特别感谢 Christian Hoock (wk1)，他贡献了浅色模式与实时配置重载功能。

---

#### 🚫 忽略错误高亮

你可以使用正则表达式跳过某些缩进错误。例如：忽略 JSDoc 的额外空格或注释行：

```jsonc
"indentRainbow.ignoreLinePatterns": [
  "/[ \t]* [*]/g",    // 匹配以 <空白><空格>* 开头的行
  "/[ \t]+[/]{2}/g"   // 匹配以 <空白>// 开头的行
]
```

跳过某些语言的缩进错误检查（例如 markdown、haskell）：

```jsonc
"indentRainbow.ignoreErrorLanguages": [
  "markdown",
  "haskell"
]
```

仅在空白字符上渲染颜色（默认 `false`）：

```jsonc
"indentRainbow.colorOnWhiteSpaceOnly": true
```

---

#### 🏗️ 构建插件

```bash
npm install
npm run vscode:prepublish
```

开发模式自动编译：

```bash
npm run compile
```

---

### 📚 总结

Indent-Rainbow 通过对缩进层级着色，使代码结构更加清晰直观；在多语言项目中，可显著减少因缩进混乱导致的语法或逻辑错误。  


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
