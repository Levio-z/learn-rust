---
tags:
  - fleeting
---
## 1. 核心观点  
### Ⅰ. 概念层

Transforms the `Option<T>` into a [`Result<T, E>`], mapping [`Some(v)`] to [`Ok(v)`] and [`None`] to [`Err(err())`].

当需要将option内容转换为Result返回的时候需要


### Ⅱ. 实现层
[[# 场景 1：把多个可能失败的操作串成管道（fallible pipeline）]]
[[# 场景 2：显式替换 if/else + match 的繁琐错误判定]]
### 案例
```rust
    let branch = rest

        .split_once(symref_prefix)

        .ok_or_else(|| anyhow::anyhow!("missing `symref=HEAD:` capability"))

        .and_then(|(_, r)| {

            r.split_whitespace()

                .next()

                .ok_or_else(|| anyhow::anyhow!("missing branch after `symref=HEAD:`"))

        })?;
```
### Ⅲ. 原理层

## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### 场景 1：把多个可能失败的操作串成管道（fallible pipeline）
### 场景 2：显式替换 if/else + match 的繁琐错误判定
```rust
match result1 {
    Ok(x) => match result2(x) {
        Ok(y) => match result3(y) {
            Ok(z) => ...
            Err(e) => return Err(e),
        }
        Err(e) => return Err(e),
    }
    Err(e) => return Err(e),
}

```
变成更简洁的：
```
result1
    .and_then(result2)
    .and_then(result3)
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
