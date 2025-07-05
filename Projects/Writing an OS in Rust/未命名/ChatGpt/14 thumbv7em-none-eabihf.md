## 🔍 三元组字段详解

|字段位置|字段名|示例值|含义|
|---|---|---|---|
|1|架构（arch）|`thumbv7em`|ARM Cortex-M4/M7，使用 Thumb 指令集|
|2|供应商（vendor）|`none`|无特定供应商（裸机环境常见）|
|3|操作系统（os）|`none`|无操作系统（`no_std` 环境）|
|4|ABI|`eabihf`|ARM EABI（硬件浮点支持）|