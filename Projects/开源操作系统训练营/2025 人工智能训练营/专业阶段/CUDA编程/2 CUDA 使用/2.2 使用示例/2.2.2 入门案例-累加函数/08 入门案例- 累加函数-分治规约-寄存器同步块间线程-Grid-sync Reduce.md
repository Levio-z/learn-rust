### 测试
```
nvcc sum_gpu/sum_gpu_07_cooperative.cu -o target/sum_gpu_07_cooperative

nsys profile -t cuda,nvtx,osrt -o target/sum_gpu_07_cooperative -f true target/sum_gpu_07_cooperative

nsys stats target/sum_gpu_07_cooperative.nsys-rep

./target/sum_gpu_07_cooperative

```
sum_gpu_07_cooperative
windows下
```
nvcc -std=c++17 -Xcompiler "/utf-8" sum_gpu/sum_gpu_07_cooperative.cu -o target/sum_gpu_07_cooperative
```
### 问题
◆ 之前用了两个 kernel 来完成跨 block 规约，避免原子操作 
◆ 为什么不能像同步 **块内/warp内** 线程一样**同步块间线程**呢？
◆ 其实…可以！

◆ 但，你得是 CUDA 9.0+ 
◆ 用 Cooperative Group！

```c++
template <typename T>

__global__ void reduce_cooperative_kernel(T* output, const T* input, size_t n) {

    // 共享内存声明（动态分配，需确保调用时足够空间）

    extern __shared__ __align__(sizeof(T)) unsigned char shared_mem_raw[];

    T* coop_smem = reinterpret_cast<T*>(shared_mem_raw);

  

    // 获取网格、线程块、线程索引等基本信息

    auto grid = cg::this_grid();

    auto block = cg::this_thread_block();

    size_t tid = threadIdx.x;          // 线程块内线程 ID

    size_t bid = blockIdx.x;           // 线程块 ID

    size_t block_size = blockDim.x;    // 线程块大小

    size_t grid_size = gridDim.x;      // 网格大小

  

    // ========== 1. 线程块级归约（Block-level reduction） ==========

    T sum = T(0);

    // 跨步遍历全局内存，避免线程发散

    for (size_t i = tid; i < n; i += block_size) {

        sum += input[i];

    }

  

    //  warp 内归约（利用 warp  shuffle 指令）

    T warp_sum = warp_reduce(sum);

  

    // 每个 warp 选一个代表线程，将 warp 结果存入共享内存

    if (tid % 32 == 0) {

        coop_smem[tid / 32] = warp_sum;

    }

    block.sync();  // 同步确保共享内存写入完成

  

    // 对共享内存中各 warp 结果，再次归约（仅前 32 线程参与）

    if (tid < 32) {

        // 读取共享内存（若超出 warp 数量则补 0）

        T block_sum = (tid < (block_size + 31) / 32)

                     ? coop_smem[tid]

                     : T(0);

        block_sum = warp_reduce(block_sum);  // 最终线程块结果

  

        // 线程块 0 号线程将结果写入全局内存，供后续全局归约

        if (tid == 0) {

            output[bid] = block_sum;

        }

    }

  

    // ========== 2. 全局级归约（Global synchronization + Final reduction） ==========

    grid.sync();  // 全局同步，确保所有线程块完成第一轮归约

  

    // 仅线程块 0 参与最终全局归约（可扩展为多线程块，此处简化）

    if (bid == 0) {

        T final_sum = T(0);

        // 类似线程块归约逻辑，遍历各线程块结果

        for (size_t i = tid; i < grid_size; i += block_size) {

            final_sum += output[i];

        }

  

        // warp 内归约

        T warp_val = warp_reduce(final_sum);

  

        // 存入共享内存（仅前 32 线程操作）

        if (tid % 32 == 0) {

            coop_smem[tid / 32] = warp_val;

        }

        block.sync();  // 同步确保写入完成

  

        // 最终全局结果（仅 0 号线程写回）

        if (tid < 32) {

            T v = (tid < (block_size + 31) / 32)

                 ? coop_smem[tid]

                 : T(0);

            T total = warp_reduce(v);

  

            // 0 号线程输出最终全局结果

            if (tid == 0) {

                output[0] = total;

            }

        }

    }

}
```
◆ 代码逻辑保持和之前的 two-pass 一样 
◆ block.sync(): __syncthreads() 
◆ grid.sync(): 跨 block 同步

◆ 但有额外需要注意的点 
	◆ cudaOccupancyMaxActiveBlocksPerMultiproce ssor(): 获取每SM对给定kernel和配置下最大的 线程块数量 
	◆ cudaDevAttrCooperativeLaunch/prop.cooperati veLaunch 确保运行环境支持 Cooperative Group 
	◆ cudaLaunchCooperativeKernel(): 发射使用 Cooperative Group 的核函数

 - int grid_size = max_blocks_per_sm * prop.multiProcessorCount;
	 - `prop.multiProcessorCount` 是 GPU 上的流式多处理器 (SM) 数量
	 - `max_blocks_per_sm`是每个 SM 可以同时容纳的最大线程块数量

- 若网格超过这个值，协同启动会直接失败（硬件无法同时容纳所有线程块）。
- 若网格远小于这个值，GPU 的 SM 资源会被浪费，性能下降。  
    因此，通过这种计算能得到一个 “安全且高效” 的网格大小上限，再结合实际数据量（如 `min(grid_size, 实际需要的线程块数)`），就能兼顾可行性和性能。

```c++
    // 计算最佳网格大小

    // prop.multiProcessorCount 是 GPU 上的流式多处理器 (SM) 数量

    // max_blocks_per_sm 是每个 SM 可以同时容纳的最大线程块数量

    int grid_size = max_blocks_per_sm * prop.multiProcessorCount;

    // 确保网格大小不会超过实际需要的数量

  

    grid_size = std::min(grid_size, static_cast<int>((n + block_size - 1) / block_size));

    grid_size = std::max(grid_size, 1);
```


```c++
    // 检查设备是否支持协同启动能力

    int can_launch = 0;

    CUDA_CHECK(cudaDeviceGetAttribute(&can_launch,

                                 cudaDevAttrCooperativeLaunch,

                                 0));  // 假设使用设备 0

    if (!can_launch) {

        std::cerr << "Error: Device does not support cooperative launches\n";

        return 1;

    }
```


```c++
    // 使用cudaLaunchCooperativeKernel启动协作核函数

    void* kernel_args[] = {&d_result, &d_input, &n};

    CUDA_CHECK(cudaLaunchCooperativeKernel(

        (void*)reduce_cooperative_kernel<float>,

        grid_size,

        block_size,

        kernel_args,

        shared_mem_size,

        0

    ));
```

![](asserts/Pasted%20image%2020250820154220.png)
观察： 
◆ 性能不错！比 Two-pass 快 
◆ 仅逊于之前的 Smem warp 规约实 现
◆ 明显，理论上这种情况应该数据量 
- 更大的时候优势会更大，看看：

◆ 换到之前的 x16 (i.e., ~16M) 观察： 
◆ Grid sync 性能成为最好而且优势 巨大！
◆ 确实数据量大的时候优势更明显

![](asserts/Pasted%20image%2020250820154607.png)
