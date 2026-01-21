---
tags:
  - note
---
## 1. 核心观点  
### Ⅰ. 概念层


### Ⅲ. 原理层

## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  

工作空间设置
[模板](../../插件/模板.md)
```
{

    // -----------------------------

    // Rust Analyzer Configuration

    // -----------------------------

    "rust-analyzer.cargo.features": "all",              // Enable all features defined in Cargo.toml

    "rust-analyzer.procMacro.enable": true,            // Enable procedural macro expansion

    "rust-analyzer.cargo.autoreload": true,           // Auto-reload Cargo project for accurate analysis

    "rust-analyzer.checkOnSave": true,                // Enable checking code on save

    "rust-analyzer.check.command": "clippy",          // Use Clippy for on-save checks

    "rust-analyzer.diagnostics.enable": true,         // Enable diagnostics

    "rust-analyzer.diagnostics.disabled": [           // Disable specific diagnostics

        "unresolved-proc-macro",                       // Ignore unresolved procedural macro warnings

        "inactive-code"                                // Ignore inactive code warnings

    ],

  

    // -----------------------------

    // Formatting and Save Settings

    // -----------------------------

    "editor.formatOnSave": true,                       // Automatically format code on save

    "editor.defaultFormatter": "rust-lang.rust-analyzer", // Use Rust Analyzer as the default formatter

    "files.autoSave": "onFocusChange",                // Auto-save files when changing focus

  

    // -----------------------------

    // Inlay Hints (Display Type Information)

    // -----------------------------

    "editor.inlayHints.enabled": "on",                // Enable inlay hints in the editor

    "rust-analyzer.inlayHints.typeHints.enable": true,       // Show type hints for variables

    "rust-analyzer.inlayHints.parameterHints.enable": true,  // Show function parameter hints

    "rust-analyzer.inlayHints.chainingHints.enable": true,   // Show intermediate types in method chains

  

    // -----------------------------

    // Optional: Enhanced Error Visibility

    // -----------------------------

    "errorLens.enabled": true,                         // Enable inline error/warning highlighting

    "errorLens.fontSize": "12px",                      // Font size for error/warning messages

    "errorLens.fontWeight": "bold",                    // Font weight

  

    // -----------------------------

    // Optional: Show TODO / FIXME Comments

    // -----------------------------

    "todo-tree.general.tags": [

        "TODO",

        "FIXME",

        "BUG"

    ]

}
```
[配置详解](../../chatgpt/配置详解.md)


## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
