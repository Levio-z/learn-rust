---
tags:
  - note
---
## 1. 核心观点  
### Ⅰ. 概念层



### Ⅱ. 应用层




### Ⅲ. 实现层

### **IV**.原理层


## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  

### 初始设置：自动补全需要的 Trie 能力清单

一个 **可用的自动补全 Trie** 至少要支持：

1. **字符串前缀匹配**（`prefix -> candidates`）
2. **可枚举子节点 / 候选**
3. **插入 / 删除（或构建期批量）**
4. **查询效率稳定（O(len(prefix)))**
5. （可选）排序、权重、频率


**crate**：`radix_trie`
### 定义与原理

- 实现的是 **Radix Trie（压缩前缀树 / Patricia Trie）**
- 连续字符压缩为一条边
- 非常适合**补全、命令索引**

### 核心 API

```rust
use radix_trie::{Trie, TrieCommon};

let mut trie = Trie::new();
trie.insert("help", 1);
trie.insert("hex", 2);

let sub = trie.get_raw_descendant("he");
for (k, _) in sub.iter() {
    println!("{}", k);
}
```

### 特点

- ✅ 原生支持前缀枚举
- ✅ 内存友好
- ✅ 与 `rustyline::Completer` 天然契合
- ❌ 不支持模糊匹配
    
#### 使用场景

- Shell / REPL
- 命令补全
- 关键字补全
    

➡ **你当前用它是“正确姿势”**

---



## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
