### 向量化

◆ 标量操作 → 向量操作 
◆ 向量化（vectorization）：一次同时操作多个标量数据 解决之前的问题： 
◆ 一次多传输数据：向量化访存，即一次同时传输（读写） 多个标量数据 → 提升访存效率
◆ 向量化本质是 SIMD (Single Instruction Multiple Data) 
◆ SIMD: 同时对一组 (向量) 数据中的每一个分别执行相同的操作从而 实现空间上的并行
### SIMT vs SIMD
![](asserts/Pasted%20image%2020250812161157.png)

### 如何向量化
◆ 想进行向量化访存，怎么操作呢？
◆ 多种方式，其中最简单的一种：内置向量化访存类型 
	◆ float2, float3, float4 …
![](asserts/Pasted%20image%2020250812161316.png)

![](asserts/Pasted%20image%2020250812161322.png)
float4 的内部实现

https://docs.nvidia.com/cuda/cuda-c-programming-guide/#built-in-vector-types

### 向量化
[8 向量化倍数和寄存器的关系](../../../chatgpt/8%20向量化倍数和寄存器的关系.md)
- 寄存器数量是有限的 → 超过上限就得用别的办法存放这些数据