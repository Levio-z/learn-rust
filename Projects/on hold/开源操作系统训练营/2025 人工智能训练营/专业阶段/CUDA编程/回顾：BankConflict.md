### 矩阵乘法
![](asserts/Pasted%20image%2020250821101704.png)

![](asserts/Pasted%20image%2020250821102544.png)
![](asserts/Pasted%20image%2020250821102612.png)
矩阵乘法（GEMM）

  

回顾：
- 之前说 GEMM 是计算密集型的算子  

分析：

- 为了方便，使用方矩阵，形状：N×N
- C（结果矩阵）中每个数（entry）对应 A 中的一行和 B 中的一列进行点积 —— O (N) 操作
- C（结果矩阵）中一共有 N×N 个数，因此计算的总复杂度为：O (N³)
- 那访存呢？一共三个矩阵，一共有 3N² 个数，因此访存的总复杂度为：O (N²)
- 计算复杂度＞访存复杂度→GEMM 理论上确实是计算密集型！

- 刚才的 \(C = A * B\) 是基础的 Matmul
- GEMM 呢？
- GEMM: \(C = \alpha A * B + \beta D\)
- A 和 B: optional transpose
- \(\alpha\) 和 \(\beta\): 标量，缩放因子，一般均为 float
- D：可以和 C 是同一个（in - place）
- 以上为基本规范，不同的地方可能会有一些其他额外的定义和功能支持

- **疑问引出**：对比一维数组加法 / 累加，提出二维矩阵（GEMM 涉及）内存存储格式的疑问
- **存储格式分类**：介绍两种基础存储方式
    - **行优先（Row - Major）**：按行顺序存储，示例矩阵存储序列为 (a1, a2, a3, a4, a5, a6, a7, a8, a9)
    - **列优先（Column - Major）**：按列顺序存储，示例矩阵存储序列为 (a1, a4, a7, a2, a5, a8, a3, a6, a9)  
        这些存储格式会影响程序访问内存效率，进而关联到 GEMM 计算性能（如行优先更适配按行遍历的计算模式，可利用缓存提升速度 ）。
### （CUDA 核函数模板，用于矩阵乘法 GEMM 计算）


```cpp
template<typename T>
__global__ void gemm_naive(const int M, const int N, const int K, float alpha, 
                           const T* A, const T* B, float beta, 
                           T* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row of C (varies per thread in y-direction)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column of C (fixed per thread in y-direction)

    if (row < M && col < N) {
        float sum = 0.0f;
        // Inner loop over k (dot product for C[row][col])
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}
```

- **参数与任务说明**：
    - `M=K=N=1024` ：矩阵维度等参数设定，这里涉及的矩阵运算中相关维度均为 1024 。
    - `二维 Block 和 Grid` ：CUDA 编程中采用二维的线程块（Block）和线程网格（Grid）模型来组织线程 。
    - `每个线程负责一个 C（结果矩阵）中数值的计算（K 次的循环计算点积）` ：每个线程对应结果矩阵 `C` 中的一个元素，通过 `K` 次循环（对应矩阵乘法中一行一列元素的点积计算）来完成该元素计算 。
    - `C 中数值按列逐步计算` ：计算 `C` 矩阵元素时，整体是按列的顺序逐步进行计算的逻辑 。
    - `可以详细看看计算动态` ：提示可进一步分析计算过程中的动态表现 。
- **性能及配置数据**：
    - `Matrix: 1024 x 1024 x 1024` ：矩阵相关维度配置 。
    - `Block size: 32 x 32` ：线程块的大小为 32×32 。
    - `Grid size: 32 x 32` ：线程网格的大小为 32×32 。
    - `Warm-up iters: 2` ：热身迭代次数，一般用于排除 GPU 初始状态等干扰，让后续性能测试更稳定 。
    - `Profile iters: 10` ：性能分析迭代次数，通过多次迭代取平均等方式统计性能 。
    - `Avg time: 7.81517 ms` ：平均每次运算时间为 7.81517 毫秒 。
    - `GFLOPS: 274.784` ：每秒浮点运算次数，衡量计算性能 ，这里约为 2747.84 亿次浮点运算每秒 。
    - `Bandwidth: 1.61006 GB/s` ：带宽，体现数据传输能力 。
    - `Verification: Passed` ：验证通过，说明计算结果正确，和理论预期或参考结果一致 。

![](asserts/Pasted%20image%2020250821104642.png)
- 按列计算
![](asserts/Pasted%20image%2020250821104752.png)

### NCU查看

![](asserts/Pasted%20image%2020250821105911.png)

![](asserts/Pasted%20image%2020250821105930.png)
![](asserts/Pasted%20image%2020250821110009.png)


![](asserts/Pasted%20image%2020250821110021.png)

![](asserts/Pasted%20image%2020250821110037.png)

![](asserts/Pasted%20image%2020250821110128.png)



![](asserts/Pasted%20image%2020250821110248.png)

![](asserts/Pasted%20image%2020250821110453.png)

![](asserts/Pasted%20image%2020250821110501.png)

![](asserts/Pasted%20image%2020250821110543.png)

### GEMM-继续优化
![](asserts/Pasted%20image%2020250821124823.png)
### 共享内存优化

![](asserts/Pasted%20image%2020250821124840.png)

![](asserts/Pasted%20image%2020250821124858.png)
#### SMEM共享内存分块技术
![](asserts/Pasted%20image%2020250821141500.png)
![](asserts/Pasted%20image%2020250821141539.png)

![](asserts/Pasted%20image%2020250821141546.png)

![](asserts/Pasted%20image%2020250821141651.png)


![](asserts/Pasted%20image%2020250821141716.png)

![](asserts/Pasted%20image%2020250821141834.png)
![](asserts/Pasted%20image%2020250821141858.png)
![](asserts/Pasted%20image%2020250821141907.png)
- 每个线程同时处理多个同列的entries

![](asserts/Pasted%20image%2020250821142008.png)


![](asserts/Pasted%20image%2020250821142131.png)
![](asserts/Pasted%20image%2020250821142142.png)
![](asserts/Pasted%20image%2020250821142201.png)

![](asserts/Pasted%20image%2020250821142315.png)




![](asserts/Pasted%20image%2020250821142329.png)
![](asserts/Pasted%20image%2020250821142334.png)

![](asserts/Pasted%20image%2020250821142350.png)


![](asserts/Pasted%20image%2020250821142450.png)

![](asserts/Pasted%20image%2020250821142528.png)

![](asserts/Pasted%20image%2020250821142646.png)

![](asserts/Pasted%20image%2020250821142652.png)


![](asserts/Pasted%20image%2020250821142818.png)


![](asserts/Pasted%20image%2020250821142839.png)


![](asserts/Pasted%20image%2020250821142905.png)


![](asserts/Pasted%20image%2020250821142934.png)



![](asserts/Pasted%20image%2020250821143001.png)


![](asserts/Pasted%20image%2020250821143153.png)
![](asserts/Pasted%20image%2020250821143329.png)


![](asserts/Pasted%20image%2020250821143400.png)


![](asserts/Pasted%20image%2020250821143448.png)


![](asserts/Pasted%20image%2020250821143608.png)

![](asserts/Pasted%20image%2020250821143634.png)
![](asserts/Pasted%20image%2020250821143745.png)






![](asserts/Pasted%20image%2020250821143756.png)






![](asserts/Pasted%20image%2020250821143806.png)



![](asserts/Pasted%20image%2020250821143855.png)


![](asserts/Pasted%20image%2020250821144305.png)

![](asserts/Pasted%20image%2020250821144457.png) 

![](asserts/Pasted%20image%2020250821144518.png)

![](asserts/Pasted%20image%2020250821144557.png)


![](asserts/Pasted%20image%2020250821144623.png)
![](asserts/Pasted%20image%2020250821144915.png)

![](asserts/Pasted%20image%2020250821144954.png)


![](asserts/Snipaste_2025-08-21_14-49-59.png)


![](asserts/Pasted%20image%2020250821145026.png)









![](asserts/Pasted%20image%2020250821145226.png)

![](asserts/Pasted%20image%2020250821145352.png)

![](asserts/Pasted%20image%2020250821145442.png)




![](asserts/Pasted%20image%2020250821145608.png)



![](asserts/Pasted%20image%2020250821145700.png)


![](asserts/Pasted%20image%2020250821145801.png)

![](asserts/Pasted%20image%2020250821160912.png)
![](asserts/Pasted%20image%2020250821161013.png)








