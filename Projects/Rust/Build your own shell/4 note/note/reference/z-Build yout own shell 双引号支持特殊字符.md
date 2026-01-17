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
### 挑战内容
#### https://app.codecrafters.io/courses/shell/stages/yt5
- 当反斜杠 `\` 出现在引号之外时，它充当转义字符。反斜杠会移除下一个字符的特殊含义，并将其视为普通字符。转义后，反斜杠本身会被移除。在这个阶段，您将实现对引号外反斜杠的支持。当反斜杠 `\` 出现在引号之外时，它充当转义字符。反斜杠会移除下一个字符的特殊含义，并将其视为普通字符。转义后，反斜杠本身会被移除。具有特殊含义的字符（例如空格、 `'` 、 `"` 、 `$` 、 `*` `?` 和其他分隔符）没有特殊含义的字符（例如 `n` 、 `t` 等普通字母）
##### 核心实现分类讨论
分类讨论相比于单引号又需要引入一个状态，非常适合用状态机实现，因此重构代码，改为状态机
##### 核心实现
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

                '\\' => match_type = MatchType::Escaping,

                _ => string.push(ch),

            },

            MatchType::SingleQuote => match ch {

                '\'' => match_type = MatchType::Default,

                _ => string.push(ch),

            },

            MatchType::DoubleQuote => match ch {

                '"' => match_type = MatchType::Default,

                '\\' => match_type = MatchType::DoubleQuoteEscaping,

                _ => string.push(ch),

            },

            MatchType::DoubleQuoteEscaping => match ch {

                '"' => {

                    string.push(ch);

                    match_type = MatchType::DoubleQuote;

                }

                '\\' => {

                    string.push(ch);

                    match_type = MatchType::DoubleQuote;

                }

                _ => {

                    string.push('\\');

                    string.push(ch);

                    match_type = MatchType::DoubleQuote;

                }

            },

            MatchType::Escaping => match ch {

                _ => {

                    string.push(ch);

                    match_type = MatchType::Default;

                }

            },

        }

    }

    if !string.is_empty() {

        params.push(string.clone());

    }

    params.into_iter()

}
```

#### https://app.codecrafters.io/courses/shell/stages/le5
- 反斜杠在[单引号](https://www.gnu.org/software/bash/manual/bash.html#Single-Quotes)内没有特殊的转义处理。单引号内的每个字符（包括反斜杠）都会被按字面意思处理。前面已经支持。
#### https://app.codecrafters.io/courses/shell/stages/gu3

在[双引号](https://www.gnu.org/software/bash/manual/bash.html#Double-Quotes)内，反斜杠仅转义某些特殊字符： `"` 、 `\` 、 `$` 、 `` ` `` 和 `newline` 。对于所有其他字符，反斜杠将按字面意思处理。

在这个阶段，我们将涵盖以下内容：

- `\"`: escapes double quote, allowing `"` to appear literally within the quoted string.  
    `\"` : 转义双引号，允许 `"` 以字面形式出现在带引号的字符串中。
- `\\`: escapes backslash, resulting in a literal `\`.  
    `\\` ：转义反斜杠，结果为字面意义上的 `\` 。

We won’t cover the following cases in this stage:  
本阶段我们将不讨论以下情况：

- `\$`: escapes the dollar sign.  
    `\$` ：用于转义美元符号。
- `` \` ``: escapes the backtick.  
    `` \` `` : 转义反引号。
- `\<newline>`: escapes a newline character.  
    `\<newline>` ：转义换行符。

Here are a few examples illustrating how backslashes behave within double quotes:  
以下是一些示例，说明反斜杠在双引号中的行为方式：


| Command  命令                        | Expected output  预期输出      |
| ---------------------------------- | -------------------------- |
| `echo "A \\ escapes itself"`       | `A \ escapes itself`       |
| `echo "A \" inside double quotes"` | `A " inside double quotes` |
|                                    |                            |

```rust
MatchType::DoubleQuote => match ch {

                '"' => match_type = MatchType::Default,

                '\\' => match_type = MatchType::DoubleQuoteEscaping,

                _ => string.push(ch),

            },

            MatchType::DoubleQuoteEscaping => match ch {

                '"' => {

                    string.push(ch);

                    match_type = MatchType::DoubleQuote;

                }

                '\\' => {

                    string.push(ch);

                    match_type = MatchType::DoubleQuote;

                }

                _ => {

                    string.push('\\');

                    string.push(ch);

                    match_type = MatchType::DoubleQuote;

                }

            },
```
#### https://app.codecrafters.io/courses/shell/stages/qj

- 已经支持，传入的一行都会被之前写的规则处理

## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
