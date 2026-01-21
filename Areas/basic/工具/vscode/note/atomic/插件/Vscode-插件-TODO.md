---
tags:
  - note
---
## 1. 核心观点  
### Ⅰ. 概念层

TODO高亮
### Ⅱ. 应用层

#### 定义与作用

**TODO Highlight** 是一个 VS Code 扩展，用于 **高亮显示代码中的注释标记**，

例如 `TODO:`、`FIXME:` 或自定义关键字。  

主要目标是提醒开发者：**代码中仍有未完成或待审查的任务**，避免忘记处理这些注释。
- **Code Lens / 高亮方式**：在编辑器中直接标记关键字，可自定义颜色、背景、边框等样式。
- **适用场景**：
    - **跟踪开发过程中的临时任务**。
    - 对团队协作的注释**进行可视化提醒**。
    - **快速定位代码中未完成或需审查的地方**

### Ⅲ. 实现层

### **IV**.原理层


## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
#### 内置关键词

- 默认关键词：`TODO:`, `FIXME:`
    
- 可自定义关键词及样式，如：
```
"todohighlight.keywords": [
  "DEBUG:",
  "REVIEW:",
  { "text": "NOTE:", "color": "#ff0000", "backgroundColor": "yellow" },
  { "text": "HACK:", "color": "#000" },
  { "text": "TODO:", "color": "red", "border": "1px solid red", "backgroundColor": "rgba(0,0,0,.2)" }
]
```


### TODO Highlight VS Code 扩展详细解释

#### 定义与作用

**TODO Highlight** 是一个 VS Code 扩展，用于 **高亮显示代码中的注释标记**，例如 `TODO:`、`FIXME:` 或自定义关键字。  
主要目标是提醒开发者：**代码中仍有未完成或待审查的任务**，避免忘记处理这些注释。

- **Code Lens / 高亮方式**：在编辑器中直接标记关键字，可自定义颜色、背景、边框等样式。
- **适用场景**：
    - 跟踪开发过程中的临时任务。
    - 对团队协作的注释进行可视化提醒。
    - 快速定位代码中未完成或需审查的地方。
        

---

#### 内置关键词

- 默认关键词：`TODO:`, `FIXME:`
- 可自定义关键词及样式，如：
    
    ```json
    "todohighlight.keywords": [
      "DEBUG:",
      "REVIEW:",
      { "text": "NOTE:", "color": "#ff0000", "backgroundColor": "yellow" },
      { "text": "HACK:", "color": "#000" },
      { "text": "TODO:", "color": "red", "border": "1px solid red", "backgroundColor": "rgba(0,0,0,.2)" }
    ]
    ```
    

---

#### 核心功能

1. **高亮关键字**
    - 支持 **颜色、背景色、边框、圆角、整行高亮** 等样式。
    - 可以通过 **正则表达式** 一次性匹配多个模式：
        ```json
        "todohighlight.keywordsPattern": "TODO:|FIXME:|\\(([^)]+)\\)"
        ```
2. **文件筛选**
    
    - **include**：指定需要扫描的文件类型，避免性能问题和二进制文件扫描。
        
        ```json
        "todohighlight.include": ["**/*.js","**/*.ts","**/*.html"]
        ```
        
    - **exclude**：排除不需要扫描的文件夹或文件，如 `node_modules`、`dist`、`.git` 等。
        
        ```json
        "todohighlight.exclude": ["**/node_modules/**","**/dist/**"]
        ```
3. **操作命令**
    - `Toggle highlight`：启用/禁用高亮。
    - `List highlighted annotations`：列出所有高亮注释，并可直接跳转到对应文件和行号。
4. **文件路径可点击问题**
    - 不同操作系统的输出路径格式不同：
        - Mac/Windows：`<path>#<line>`
        - Linux：`<path>:<line>:<column>`
    - 可通过设置 `todohighlight.toggleURI: true` 切换路径格式，使其可点击跳转。
        
---

#### 配置选项详解

|设置|类型|默认值|说明|
|---|---|---|---|
|`todohighlight.isEnable`|boolean|true|是否启用高亮|
|`todohighlight.isCaseSensitive`|boolean|true|是否区分大小写|
|`todohighlight.keywords`|array|N/A|自定义关键字和样式|
|`todohighlight.keywordsPattern`|string|N/A|使用正则匹配关键字，优先级高于 keywords|
|`todohighlight.defaultStyle`|object|N/A|自定义关键字默认样式|
|`todohighlight.include`|array|常用 JS/TS/HTML/CSS 文件|扫描的文件类型|
|`todohighlight.exclude`|array|常用排除目录|排除扫描的文件/目录|
|`todohighlight.maxFilesForSearch`|number|5120|最大搜索文件数，避免性能问题|
|`todohighlight.toggleURI`|boolean|false|切换输出路径格式使其可点击|

---

#### 示例配置

```json
{
  "todohighlight.isEnable": true,
  "todohighlight.isCaseSensitive": true,
  "todohighlight.keywords": [
    "DEBUG:",
    "REVIEW:",
    { "text": "NOTE:", "color": "#ff0000", "backgroundColor": "yellow" },
    { "text": "HACK:", "color": "#000" },
    { "text": "TODO:", "color": "red", "border": "1px solid red", "backgroundColor": "rgba(0,0,0,.2)" }
  ],
  "todohighlight.keywordsPattern": "TODO:|FIXME:|\\(([^)]+)\\)",
  "todohighlight.include": ["**/*.js","**/*.ts","**/*.html"],
  "todohighlight.exclude": ["**/node_modules/**","**/dist/**"]
}
```


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
