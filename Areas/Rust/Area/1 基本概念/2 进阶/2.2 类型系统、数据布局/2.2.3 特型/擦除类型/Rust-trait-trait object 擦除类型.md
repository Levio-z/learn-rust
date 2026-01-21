---
tags:
  - permanent
---
## 1. 核心观点  
### Ⅰ. 概念层

但返回要求一个类型使用box擦除类型
- [案例](#案例)
### Ⅱ. 应用层




### Ⅲ. 实现层

### **IV**.原理层


## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### 案例
```RUST
fn get_problem_id(id: &u32) -> String {
    fs::read_dir("./src/problem")
        .map(|rd| {
            Box::new(rd.filter_map(|x| x.ok()).filter_map(|x| {
                let file_name = x.file_name().into_string().ok()?;
                if file_name.starts_with(&format!("p{:04}", id)) {
                    Some(file_name.strip_suffix(".rs").unwrap().to_string())
                } else {
                    None
                }
            })) as Box<dyn Iterator<Item = String>>
        })
        .unwrap_or_else(|_| Box::new(std::iter::empty()))
        .next()
        .unwrap_or_default()
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
 
  
