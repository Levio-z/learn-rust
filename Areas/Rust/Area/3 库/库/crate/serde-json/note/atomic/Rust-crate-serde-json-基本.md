---
tags:
  - fleeting
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
### 基本案例
iter：（key，value）=>Value 一个接送记录
```rust
    for result in reader.records() {

        let record = result.context("Failed to deserialize record")?;

        let json_value = headers.iter().zip(record.iter()).collect::<Value>();

        println!("{:?}", json_value);

        vec.push(json_value);

    }

    let json_data =

        serde_json::to_string_pretty(&vec).context("Failed to serialize records to JSON")?;

    std::fs::write(output, json_data).context("Failed to write output file")?;
```

## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
