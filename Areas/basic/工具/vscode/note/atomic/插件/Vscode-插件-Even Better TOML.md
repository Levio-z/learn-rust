### 🧩 简介

这是一个由 **Taplo** 提供支持的 **TOML 语言支持扩展**。  
目前处于预览阶段，可能存在一些错误，甚至可能崩溃。  
如果你遇到任何问题，请在 GitHub 上提交 issue。

---

### 📚 功能特性

#### TOML 1.0.0 版本支持

该扩展目前支持 **TOML v1.0.0**，并将在未来尽力支持所有 TOML 版本。

---

#### 语法高亮（Syntax Highlighting）

为 TOML 文件提供基于 **TextMate 语法规则** 的语法高亮显示。

**示例：**  
为键、值、表头、数组等元素提供直观的颜色区分。

---

#### 扩展语法颜色（Additional Syntax Colors）

扩展为 **数组头部（array headers）** 和 **表数组（arrays of tables）** 定义了额外的颜色范围（scope）。

为了让它们与普通键（keys）区别开来，你可以手动设置专属颜色。  
另外，建议你为日期（date）和时间（time）类型单独设置颜色，因为大多数主题默认没有为它们着色。

**示例：**
- 自定义 Dark+ 主题的颜色配置
- 扩展后的颜色高亮效果

---

#### 语义高亮（Semantic Highlighting）

可以通过设置启用针对 **内联表（inline tables）** 和 **数组（arrays）** 的语义键高亮。  
若要获得更好的显示效果，需同时配置扩展颜色。

---

#### 验证（Validation）

自动检查 TOML 文件格式和语法错误。

**示例：**  
错误会在编辑器中直接高亮提示。

---

#### 折叠（Folding）

支持以下内容的折叠功能：

- 数组
    
- 多行字符串
    
- 顶层表
    
- 注释
    

---

#### 符号树与导航（Symbol Tree and Navigation）

即使表（table）顺序混乱，也能正确生成符号树。  
你可以方便地在不同表、键之间导航。

---

#### 重构（Refactors）

支持 **重命名（Rename）** 功能。  
可通过命令面板或快捷键重命名键名。

---

#### 格式化（Formatting）

默认格式化风格较为保守，可在设置中启用更多特性。  
如果你希望增加新的格式化选项，可在 GitHub 上提交 issue 请求。

---

#### 使用 JSON Schema 进行补全与验证

支持以下功能：

- 自动补全（Completion）
    
- 悬停提示（Hover Text）
    
- 链接跳转（Links）
    
- 校验（Validation）
    

可以在配置项 **`evenBetterToml.schema.associations`** 中为文档 URI 关联 Schema。  
你可以：

- 提供自定义 Schema
    
- 或使用 **JSON Schema Store** 中的现有 Schema
    

详细说明请参阅扩展文档。

---

#### 命令（Commands）

该扩展提供了方便的 **JSON ↔ TOML 相互转换** 命令。

---

#### 配置文件（Configuration File）

该扩展支持并自动识别 **Taplo CLI 的配置文件**，  
会自动在工作区根目录查找，也可以在 VS Code 设置中手动指定路径。

---

#### 致谢（Special Thanks）

- 致谢 **@GalAster** 和 **@be5invis**，允许使用他们的 TextMate 语法文件
    
- 致谢所有贡献者
    
- 感谢每一位使用本扩展的用户 ❤️
    

---

是否希望我帮你总结一下这个扩展最有用的部分（比如哪些功能值得启用、如何配合 JSON Schema 使用）？


