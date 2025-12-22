---
tags:
  - reference
---
## 1. 核心观点  
### Ⅰ. 概念层



### Ⅱ. 实现层



### Ⅲ. 原理层

## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  

### Single Quotes  单引号

[单引号](https://www.gnu.org/software/bash/manual/bash.html#Single-Quotes) （ `'` ）会禁用其内字符的所有特殊含义。单引号内的每个字符都将按字面意思处理。
- 单引号内的字符（包括转义字符和特殊字符，如 `$` 、 `*` 或 `~` ）失去其特殊含义，并被视为普通字符。
- 单引号内的连续空白字符（空格、制表符）将被保留，不会被折叠或用作分隔符。
- 相邻的带引号的字符串会被连接起来形成一个参数。

在 shell 语法中， [双引号](https://www.gnu.org/software/bash/manual/bash.html#Double-Quotes) ( `"` ) 内的大多数字符都会被按字面意思处理。但是，双引号允许解释某些特殊字符（例如 `$` 表示变量， `\` 表示转义），我们将在后面的章节中介绍这些例外情况。

在此阶段，您的 shell 在解析双引号时必须应用以下规则：

- 必须保留连续的空白字符（空格、制表符）。
- 通常用作分隔符或特殊字符的字符在双引号内会失去其特殊含义，并按字面意思处理。
- 相邻的双引号字符串会被连接起来形成一个参数。

### 核心实现分类讨论
分类讨论相比于单引号又需要引入一个状态，非常适合用状态机实现，因此重构代码，改为状态机

```
enum MatchType {
	//默认
    Default,
	//双引号中
    DoubleQuote,
	//单引号中
    SingleQuote,
	//默认转义
    Escaping,
	//双引号中转义
    DoubleQuoteEscaping,
}
```
### 核心实现
```rust
fn split_quotes(line: &str) -> impl Iterator<Item = String> {

    let mut params = Vec::new();

    let mut string = String::new();

    let mut match_type = MatchType::Default;

    for ch in line.chars() {

        match match_type {

            MatchType::Default => match ch {

                ch if ch.is_whitespace() => {

                    if !string.is_empty() {

                        params.push(string.clone());

                        string = String::new();

                    }

                    continue;

                }

                '\'' => match_type = MatchType::SingleQuote,

                '"' => match_type = MatchType::DoubleQuote,

                _ => string.push(ch),

            },

            MatchType::SingleQuote => match ch {

                '\'' => match_type = MatchType::Default,

                _ => string.push(ch),

            },

            MatchType::DoubleQuote => match ch {

                '"' => match_type = MatchType::Default,

                _ => string.push(ch),

            },

        }

    }

    if !string.is_empty() {

        params.push(string.clone());

    }

    params.into_iter()

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
- [ ] 深入阅读 xxx  
- [ ] 验证这个观点的边界条件  
