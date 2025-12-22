sum_gpu_04_reduce_warp_shfl.cu
### 测试
数据大小：const size_t SIZE = 1 << 20; // 元素总数 half个数
```
nvcc sum_gpu/sum_gpu_04_reduce_warp_shfl.cu -o target/sum_gpu_04_reduce_warp_shfl

nsys profile -t cuda,nvtx,osrt -o target/sum_gpu_04_reduce_warp_shfl -f true target/sum_gpu_04_reduce_warp_shfl

nsys stats target/sum_gpu_04_reduce_warp_shfl.nsys-rep

./target/sum_gpu_04_reduce_warp_shfl

```
### 继续优化


◆ 还有没有能继续优化的方法？ 
观察： 
◆ 树状规约一个问题：越往后**活跃的 线程越少，闲置的线程越多，利用 率下降** 
◆ 原子操作现在每个 block 一次，能 不能继续减少？

![](asserts/Pasted%20image%2020250820110604.png)
```c++
template <typename T>

__global__ void reduce_warp_shfl_register_kernel(T* output, const T* input, size_t n) {

    size_t tid = threadIdx.x;

    size_t i = tid;

  

    T sum = 0;

    // 线程束内跨线程加载数据累加

    for (; i < n; i += blockDim.x * gridDim.x) {

        sum += input[i];

    }

  

    // 基于 warp shuffle 指令的归约操作

    for (int offset = 16; offset > 0; offset >>= 1) {

	        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    }

  

    // 每 32 个线程（一个 warp）的首线程将结果原子加到全局输出

    if (tid % 32 == 0) {

        atomicAdd(output, sum);

    }

}
```
### 代码说明

1. **模板与核函数定义**：
    - 定义了模板函数 `reduce_warp_shfl_register_kernel`，通过 `__global__` 关键字声明为 CUDA 核函数，可在 GPU 上并行执行。`T` 为模板参数，用于适配不同数值类型的归约计算。
    - 核函数参数 `output` 是指向结果存储位置的设备端指针，`input` 是指向输入数据的设备端指针，`n` 表示输入数据的总元素个数 。
2. **数据加载与初步累加**：
    - 每个线程根据自身线程索引 `tid`，以 `blockDim.x * gridDim.x`（线程块数量乘以每个线程块内线程数，即全局线程步长 ）为间隔遍历数据，将对应位置的 `input` 元素累加到 `sum` 中，实现初步的数据并行加载与局部累加 。
3. **基于 Warp Shuffle 的归约**：
    - 利用 `__shfl_down_sync` 函数（CUDA 中线程束内线程间数据交换指令 ），从偏移量 `16` 开始，不断折半偏移量，将线程束内不同线程的 `sum` 值进行交换并累加，逐步完成线程束内的归约操作，把一个线程束（通常 32 个线程 ）内所有线程的计算结果合并到线程束中第一个线程的 `sum` 里 。
4. **结果原子写回全局内存**：
    - 当线程索引 `tid` 是 `32` 的倍数（即每个线程束的首个线程 ）时，调用 `atomicAdd` 函数，以原子操作的方式将该线程束归约后的 `sum` 值加到全局内存的 `output` 地址对应位置，避免多个线程同时写 `output` 引发数据竞争问题 。


### Warp规约优化
◆ 多了个奇怪的东西 
◆__shfl_down_sync(): **线程束洗牌函数 （Warp Shuffle Function）**
- 第一个mark，线程范围
- 从 lane_id + offset 处取 sum 的值 
- [10 __shfl_down_sync](../../../chatgpt/10%20__shfl_down_sync.md)

特点
- 使用 register 无需共享内存和同步 → 高效 
-  超出边界反正本值 → 无需担心边界问题
	- 当通过 `lane_id + offset` 这个索引去读取 `sum` 的值时，如果计算出的索引超出了 `sum` 所在数据结构（比如数组）的有效边界范围，这种情况下依然会返回一个合理的 “默认值”（或符合预期的基础值），而不会因为索引越界导致程序出错（如崩溃、抛出异常等）。
![](asserts/Pasted%20image%2020250820125500.png)

0.25提升到1.5

◆ 相比于之前的版本，使用 __shfl_down_sync() 后 Roofline 显示 AI 明显提高了很多 
◆ 原因？因为__shfl_down_sync() 使得整理计算量增加，AI 增加
![](../../../asserts/Pasted%20image%2020250813154055.png)

观察： 
◆ 确实有明显提升！ 
◆ 但跟目前最快的 smem-tree 规约实 现比还是差距很大，原因？
◆ 因为我们没对 warp 间规约优化！→ 将原来的 smem 手动树状规约换成 warp-shuffle 规约