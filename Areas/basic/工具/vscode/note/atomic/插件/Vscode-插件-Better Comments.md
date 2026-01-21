## 1. 核心观点  
### Ⅰ. 概念层

- ！ **Alerts（警告）**
- ？ **Queries（疑问）**
- TODO:**TODOs（待办事项）**
- * **Highlights（重点标注）**
- //// **注释掉的代码** 也可以以不同样式显示，明确提示这段代码不应存在
- 🎨 你还可以在设置中自定义其他类型的注释样式
![注释示例](https://github.com/aaron-bond/better-comments/raw/HEAD/images/better-comments.PNG)

### Ⅱ. 应用层




### Ⅲ. 实现层

### **IV**.原理层


## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  

**Better Comments** 扩展帮助你在代码中创建更具可读性、更人性化的注释。  
通过它，你可以将注释分类并以不同的样式高亮显示，例如：





---

## ⚙️ 配置说明

Better Comments 的配置可以在 **用户设置 (User Settings)** 或 **工作区设置 (Workspace Settings)** 中完成。

### 1️⃣ `better-comments.multilineComments`

控制多行注释是否启用样式化。

```json
"better-comments.multilineComments": true
```

- **true**：多行注释将根据标签样式化显示。
    
- **false**：多行注释将以普通文本显示，不应用颜色或格式。
    

---

### 2️⃣ `better-comments.highlightPlainText`

控制是否在纯文本文件中识别注释标记。

```json
"better-comments.highlightPlainText": false
```

- **true**：当注释标签（默认 `! * ? //`）位于行首时，也会被识别并高亮。
    
- **false**：纯文本文件中的注释不会被高亮。
    

---

### 3️⃣ `better-comments.tags`

定义注释标签与对应样式。  
你可以修改默认 5 个标签的颜色或添加更多自定义标签。

```json
"better-comments.tags": [
  {
    "tag": "!",
    "color": "#FF2D00",         // 红色：警告
    "strikethrough": false,
    "underline": false,
    "backgroundColor": "transparent",
    "bold": false,
    "italic": false
  },
  {
    "tag": "?",
    "color": "#3498DB",         // 蓝色：问题、疑问
    "strikethrough": false,
    "underline": false,
    "backgroundColor": "transparent",
    "bold": false,
    "italic": false
  },
  {
    "tag": "//",
    "color": "#474747",         // 灰色：注释掉的代码
    "strikethrough": true,
    "underline": false,
    "backgroundColor": "transparent",
    "bold": false,
    "italic": false
  },
  {
    "tag": "todo",
    "color": "#FF8C00",         // 橙色：待办事项
    "strikethrough": false,
    "underline": false,
    "backgroundColor": "transparent",
    "bold": false,
    "italic": false
  },
  {
    "tag": "*",
    "color": "#98C379",         // 绿色：重点或备注
    "strikethrough": false,
    "underline": false,
    "backgroundColor": "transparent",
    "bold": false,
    "italic": false
  }
]
```

---

## 🧭 总结

**Better Comments** 的核心价值在于：  
让注释成为“可视化信息层”，让开发者一眼识别出：

- 哪些是警告
    
- 哪些是问题
    
- 哪些是 TODO
    
- 哪些是重点提示  
    从而提升团队协作和代码可维护性。
    

> **推荐做法：**
> 
> - 将标签规范化，例如：  
>     `// ! 警告`、`// ? 疑问`、`// todo 待完善`、`// * 核心逻辑`
>     
> - 在 `.vscode/settings.json` 中统一配置团队的注释风格。


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
