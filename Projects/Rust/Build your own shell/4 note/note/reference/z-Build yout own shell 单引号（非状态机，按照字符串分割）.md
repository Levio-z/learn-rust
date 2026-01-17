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

### 核心实现分类讨论
声明一个可变变量string，用于存储单个参数。

- 不在引号中
	- 遇到空格
		- 将小段的内容添加到了参数的末尾，将该变量clone添加到vec中，此时获取了一个
	- 遇到`'`
		- 标记在引号中，将引号前面的内容添加到string末尾
- 在引号中
	- 继续遇到引号，将两段引号的内容添加到string末尾
### 核心实现
```rust
fn split_quotes(line: &str) -> impl Iterator<Item = String> {

    let mut params = Vec::new();

    let mut start = 0;

    let mut string = String::new();

    let mut is_quote = false;

    line.chars().enumerate().for_each(|(i, c)| {

        if !is_quote {

            if c.is_whitespace() {

                if start < i {

                    string.push_str(&line[start..i]);

                }

                if !string.is_empty() {

                    params.push(string.clone());

                    string = String::new();

                }

                start = i + 1;

            } else if c == '\'' {

                is_quote = true;

                if start < i {

                    string.push_str(&line[start..i]);

                }

                start = i + 1;

            }

        } else if is_quote && c == '\'' {

            is_quote = false;

            if start < i {

                string.push_str(&line[start..i]);

            }

            start = i + 1;

        }

    });

    string.push_str(&line[start..]);

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
 
  
