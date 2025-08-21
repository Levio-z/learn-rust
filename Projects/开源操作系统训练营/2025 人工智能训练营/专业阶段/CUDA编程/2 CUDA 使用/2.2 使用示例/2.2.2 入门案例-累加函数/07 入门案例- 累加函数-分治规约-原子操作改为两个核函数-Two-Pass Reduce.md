sum_gpu_06_two_reduce


之前的另一个问题： 
◆ 原子操作现在每个 block 一次，能 不能继续减少？ 
答案：
◆ 还真能！

◆ 原子操作目前的作用是跨 Block 规约 

◆ 或许可以分级？两个核函数： 
◆ 第一个是目前的 Block 内规约 
◆ 第二个在第一个的基础上跨 Block 规约

◆ 理论可行，实践试试！


问题： 
◆ 第一个 kernel 的结果需要存到全局内存 → 需要分配一个临时数组 注意： 
◆ 第二个 kernel 的 gridSize 是 1 
- 只有一个block
◆ 为什么？
sum_gpu_06_two_reduce
### 测试
```cpp
nvcc sum_gpu/sum_gpu_06_two_reduce.cu -o target/sum_gpu_06_two_reduce

nsys profile -t cuda,nvtx,osrt -o target/sum_gpu_06_two_reduce -f true target/sum_gpu_06_two_reduce

nsys stats target/sum_gpu_06_two_reduce.nsys-rep

./target/sum_gpu_06_two_reduce

```


第一个核函数
◆ 第一个 Kernel 逻辑和之前基本相同 ◆ 最后的块内规约从 原子操作跨 block 规约 → 直接赋值到中间数组
第二个核函数
◆ 和第一个 Kernel 逻辑基本相同
◆ Grid Stride Loop → Block Stride Loop 
◆ 原子操作数：0 ！ 
◆ 测测性能如何
![](asserts/Pasted%20image%2020250820145934.png)

观察： 
◆ 性能反而降低了… 分析原因： 
◆ 因为分成了两个 kernel，有 kernel 发射、额外计算、访存以及隐式同步（第二个核函数需要第一个核函数完成才行）的开销！抵消了想要减小原子操作带来的收益。


### 特性对比

Smem Atomic Reduce
Pros： 
◆ 代码简单，一个 kernel 
Cons:
◆ 需要原子操作

Two-Pass Reduce
Pros： 
◆ 无需原子操作 
Cons: **
◆ 额外核函数发射、隐式同步、计算和访存（额外中间数组）开销**

结论：**数据量大、原子操作（半精度）占时高时用 Two-Pass**
>半精度而言原子操作
