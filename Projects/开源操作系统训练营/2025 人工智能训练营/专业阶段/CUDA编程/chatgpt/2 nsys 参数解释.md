```
nsys profile -t cuda,nvtx,osrt -o add_cuda -f true ./add_cuda
```
>- 运行 `./add_cuda` 程序
- 同时采集 CUDA 相关 API 和内核执行数据
- 采集程序中使用 NVTX 插入的标记事件（如有）
- 采集操作系统线程和调度相关事件
- 结果保存为 `add_cuda.qdrep` 等文件，方便用 NVIDIA Nsight Systems GUI 工具打开进行分析
- 如果已有 `add_cuda.qdrep` 文件，则自动覆盖

| 参数                  | 说明                                                                                                                                                                 |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `nsys profile`      | 启动 Nsight Systems 的性能分析，会启动指定的程序并收集性能数据。                                                                                                                           |
| `-t cuda,nvtx,osrt` | 指定采集的事件类型：  <br>- `cuda`：CUDA API 调用和 GPU 活动（kernel、memcpy 等）。  <br>- `nvtx`：用户插入的 NVTX 标记（NVIDIA Tools Extension，可用于代码区域标记）。  <br>- `osrt`：操作系统运行时事件（线程调度、进程行为等）。 |
| `-o add_cuda`       | 指定输出文件名，分析结果保存为 `add_cuda.qdrep` 和 `add_cuda.sqlite` 等文件，方便后续用 Nsight Systems GUI 打开查看。                                                                            |
| `-f true`           | 开启“强制覆盖”模式，表示如果已有同名文件，自动覆盖。避免因文件存在而报错。                                                                                                                             |
| `./add_cuda`        | 需要被分析的可执行程序，当前目录下的 `add_cuda`。                                                                                                                                     |
