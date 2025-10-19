sum_gpu_05_reduce_warp_shfl_plus.cu
### 测试
数据大小：const size_t SIZE = 1 << 20; // 元素总数 half个数
```
nvcc sum_gpu/sum_gpu_05_reduce_warp_shfl_plus.cu -o target/sum_gpu_05_reduce_warp_shfl_plus

nsys profile -t cuda,nvtx,osrt -o target/sum_gpu_05_reduce_warp_shfl_plus -f true target/sum_gpu_05_reduce_warp_shfl_plus

nsys stats target/sum_gpu_05_reduce_warp_shfl_plus.nsys-rep

./target/sum_gpu_05_reduce_warp_shfl_plus

```


```c++
template<typename T>

__device__ T wrap_reduce(T val){

#pragma unroll

     // 基于 warp shuffle 指令的归约操作

     for (int offset = 16; offset > 0; offset >>= 1) {

        val += __shfl_down_sync(0xFFFFFFFF, val, offset);

    }    

    return val;

}

  

template <typename T>

__global__ void reduce_warp_shfl_register_kernel(T* output, const T* input, size_t n) {

    extern __shared__ T smem[];

    // 当前线程在一个 block 内的一维索引。

    size_t tid = threadIdx.x;

    size_t idx = blockIdx.x * blockDim.x + tid;  

  

    T sum = 0;

  

    // 线程束内跨线程加载数据累加

    for (size_t i = idx; i < n; i += blockDim.x * gridDim.x) {

        sum += input[i];

    }

    // 所有线程按照线程束规约

    T wrap_sum = wrap_reduce(sum);

  

    // 线程束规约结果移动到共享内存

    if (tid % 32 == 0){

        smem[tid /32] = wrap_sum;

    }

    __syncthreads();

    // 一个 block 最多 32 个 warp → 最多 32 个部分和 → 最多需要 32 个线程来做跨 warp 的归约。

    if (tid < 32){

        // 一个 block 里可能有很多 warp,如果 blockDim.x = 128，那么就有 128 / 32 = 4 个 warp。

        T block_sum  = ( tid < (blockDim.x + 31) /32 )? smem[tid] : T(0);

        // 对wrap内规约

        block_sum = wrap_reduce(block_sum);

        // 每个块规约

        if (tid==0){

            atomicAdd(output,block_sum);

        }

    }

}
```


◆ 抽象出一个使用 __shfl_down_sync() 的 warp_reduce() 函数 
◆ `#pragma unroll: 暗示编译器进行循环展开 `
- 对于这样情况，效率会更高
◆ 前面跟之前一样，grid stride 循环 + warp 内 规约 
- 步骤
	- block所有线程按照线程束进行规约，每32都规约到第一个，使用共享内存记录
	- 线程束个数一定是32，正好是一个线程束，可以继续使用线程束规约，每个block内的规约结束
	- 对于block之间使用原子操作规约
◆ 块内 Warp 间规约用 smem 进行一次 warp_reduce，然后将结果加到最终输出 
**◆ 原子操作数：1/block**

![](asserts/Pasted%20image%2020250820140717.png)