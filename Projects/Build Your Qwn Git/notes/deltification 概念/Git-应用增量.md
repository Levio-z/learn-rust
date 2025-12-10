---
tags:
  - note
---
## 1. 核心观点  

将增量应用于基本对象以重建缺失的对象
## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  

### 1. 增量应用（Applying Deltas）概念

在 Git 中，**delta 对象**只存储了与基对象的差异，因此在解码 packfile 时，需要 **应用增量** 来重建完整的目标对象。

整个过程大致如下：

```
base object + delta → reconstructed object
```

delta 数据结构为：

```
[source length][target length][instruction stream]
```

- **source length**：基对象大小（可变长度整数）
- **target length**：目标对象大小（可变长度整数）
- **instruction stream**：一系列“复制”或“插入”指令，用于重建目标对象
  
> source/target 长度主要用于 **错误检查**，确保生成的对象大小正确。

---

### 2. 指令类型

#### (1) 插入（Insert, MSB=0）

- **规则：** MSB = 0 表示插入指令
- **长度：** 指令本身低 7 位 = 待插入字节数（最大 127）
- **操作：**
    - 从 delta 数据流读取指定位数的字节
    - 直接写入目标对象
- **示例：**
    ```
    指令: 01001011 → 0x4B = 75
    动作: 从 delta 流读取接下来的 75 个字节写入输出
    ```
    

#### (2) 复制（Copy, MSB=1）

- **规则：** MSB = 1 表示复制指令
- **功能：** 从基对象中复制连续字节到目标对象
- **参数：**
    1. **偏移量（offset）**：基对象起始字节位置
    2. **长度（length）**：复制字节数
- **压缩编码机制：**
    - Git 只存储 **非零字节**
    - 指令字节中最后 4 位表示 **offset 字节数**
    - 指令中间 3 位表示 **length 字节数**
    - 小端排列（least significant byte first）
        
- **示例：**
    
    ```
    指令低 4 位: 1010 → 表示使用第 1、3、5、7 个字节构造 offset
    下两字节: 11010111 01001011
    偏移计算: 01001011 00000000 11010111 00000000 = 1,258,346,240
    ```
    

> 这种设计可高效表示大偏移量而节省空间，例如 32 位偏移量仅用几个字节表示。

---

### 3. 总体执行流程

伪代码如下：

```text
read delta_source_length
read delta_target_length
target = []

while not end_of_delta_stream:
    read instruction_byte
    if MSB == 0:        # insert
        length = low7bits(instruction_byte)
        data = read(length)
        target.append(data)
    else:               # copy
        offset = decode_offset(instruction_byte, delta_stream)
        length = decode_length(instruction_byte, delta_stream)
        target.append(base[offset:offset+length])
```

- **插入**直接把 delta 数据写入
- **复制**从基对象中拷贝指定偏移与长度
- 最终生成的 `target` 长度应等于 `delta_target_length`
---

### 4. 设计原理与优势

1. **空间优化**
    - 只存储差异内容，重复数据用复制指令复用
    - 偏移量和长度用压缩编码节省空间
2. **支持大对象**
    - 可变长度整数 + 分散字节存储，使得偏移量可达数 GB
3. **流式解码**
    - 只需基对象 + delta 数据即可逐条指令生成目标对象
    - 无需随机访问整个 packfile

---

### ✅ 总结

**总结：**

- Delta 开头是 source/target 长度，用于错误检查
- 指令分为复制（MSB=1）和插入（MSB=0）
- 复制指令参数压缩存储（非零字节 + 指定位数 + 小端排列）
- 插入指令直接写入指定字节
- 应用所有指令即可重建完整对象
    

**学习方法论：**

- 用实际 packfile 中的 delta 对象练习解析每条指令
- 对比基对象与生成对象，理解复制/插入指令作用
- 编写解析器将 delta 数据转换为完整对象
    

**练习题：**

1. 给出一条复制指令，说明如何解码 offset 与 length
2. 给出一条插入指令，计算应插入的字节数
3. 手动执行 2~3 条 delta 指令，生成目标对象
4. 解释为什么偏移量压缩比存储完整 32 位整数更节省空间
    

**高价值底层知识点：**

- Delta 指令压缩编码机制
- 基对象复用原理
- 可变长度整数与指令位域解析
- 流式增量解码的空间效率

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
