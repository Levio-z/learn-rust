---
tags:
  - note
---
## 1. 核心观点  
### Ⅰ. 概念层

**Lexer（词法分析）** 的职责是：

> **把“字符流”转换为“有边界、有类型的记号（Token）”**

它**只关心字符如何组合成最小语义单元**，不关心这些单元“是什么意思”。


### Ⅱ. 实现层



### Ⅲ. 原理层

## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### 初始设置：Parser 的定义

**Parser（语法分析）** 的职责是：

> **把“Token 序列”组合成“有结构、有语义的语法树（AST）”**

Parser **不处理字符，只处理 Token 类型**。

---

### 初始设置：Shell Parser 关心什么？

Parser 关心的是**命令语义结构**：

- 哪些 `Word` 组成 argv
    
- `|` 如何拆分 pipeline
    
- `>` / `>>` 是“重定向语义”
    
- `2>` 是 stderr 重定向
    
- `>` 后面必须跟一个 `Word`
    

例如：

```rust
Word("echo"),
Word("hello"),
RedirectOut,
Word("out.txt")
```

Parser 的产出是：

```rust
CommandSpec {
    argv: ["echo", "hello"],
    redirects: [
        Redirect::Stdout {
            path: "out.txt",
            append: false
        }
    ]
}
```

---

### 初始设置：redirect 规则只存在于 Parser

**redirect 是“命令语义规则”**，不是字符规则：

- `>` 是 stdout
    
- `2>` 是 stderr
    
- `>>` 是 append
    
- `>` 后面必须有文件名
    
- `echo >` 是语法错误
    

这些问题的本质是：

> **“Token 如何组合成合法的命令结构”**

👉 这是 Parser 的职责。

Lexer 只需产出：

```rust
RedirectOut
RedirectAppendOut
```

至于：

- 合不合法？
    
- 跟哪个文件？
    
- 是否允许多次？
    

Lexer 完全不关心。




## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
