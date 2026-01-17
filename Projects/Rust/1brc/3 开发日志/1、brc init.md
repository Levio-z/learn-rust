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
##### perf
```
perf record -g cargo run --release
perf report -g
```

发布模式编译，看不到任何符号


#### 1. `lto = "fat"`

- **含义**：`fat` 是 `full` 的别名（在 Rust 1.59 + 版本中统一，早期版本可能用`fat`），全称是 Link Time Optimization（链接时优化）。
- **作用**：编译器会在链接阶段对整个项目的所有代码（包括依赖库）进行全局优化，能显著减小二进制文件体积、提升运行性能，但会**大幅增加编译时间**（尤其是大型项目）。
- **适用场景**：生产环境发布、对性能 / 体积要求高的场景；开发阶段不建议开启，会拖慢调试效率。

#### 2. `panic = "abort"`

- **含义**：指定程序发生`panic!`（运行时恐慌）时的行为为 “直接终止进程”。
- **对比默认行为**：Rust 默认`panic = "unwind"`（展开调用栈，清理资源后退出），而`abort`会立即终止，不做任何清理。
- **作用**：
    
    - 减小二进制文件体积（无需包含栈展开的相关代码）；
    - 提升程序终止速度，但可能导致资源泄漏（如未关闭的文件、网络连接）。
    
- **适用场景**：对体积极致优化、且程序 panic 后无需优雅清理资源的场景（如嵌入式、极简工具）。

#### 3. `debug = true`

- **含义**：在`release`（发布）模式下依然生成调试信息。
- **默认行为**：`release`模式下`debug = false`（不生成调试信息，减小体积），`debug`模式下`debug = true`。
- **作用**：
    
    - 允许用调试器（如 gdb、lldb）调试发布版本的程序，定位生产环境的问题；
    - 会略微增大二进制文件体积，但不影响运行性能（因为优化依然开启）。
    
- **适用场景**：需要排查发布版本 Bug、但又不想放弃编译优化的场景。

### 完整示例（可直接复制到 Cargo.toml）

toml

```
[profile.release]
# 链接时全量优化（提升性能/减小体积，增加编译时间）
lto = "fat"
# panic时直接终止进程（减小体积，无资源清理）
panic = "abort"
# 保留调试信息（方便调试发布版本）
debug = true
# 可选：补充常用的发布模式优化配置
opt-level = 3  # 最高级别的编译优化（默认也是3）
strip = true    # 剥离符号表（进一步减小体积，若开启debug=true则不建议加）
```

```


________________________________________________________
Executed in   51.13 secs    fish           external
   usr time   49.41 secs  486.00 micros   49.41 secs
   sys time    0.70 secs    0.00 micros    0.70 secs

```
##### 创建cargo config文件
#### 1. `-Ctarget-cpu=native`

- **核心含义**：告诉 Rust 编译器，针对当前编译机器的 CPU 架构（本机 CPU）进行指令集优化，而不是编译成兼容所有 CPU 的通用二进制文件。
- **通俗解释**：
    
    - 默认情况下，Rust 会编译出兼容目标架构（如 x86_64）所有通用 CPU 的代码，只使用基础指令集；
    - 加上这个参数后，编译器会检测你当前电脑的 CPU 型号（比如 Intel i7-12700H、AMD Ryzen 7 5800X），启用该 CPU 支持的所有高级指令集（如 AVX2、SSE4.2、AES-NI 等）。

- **实际影响**：
    
    ✅ **优点**：程序运行性能提升（尤其是计算密集型任务，如数值运算、加密解密），能充分利用本机 CPU 的硬件特性；
    
    ❌ **缺点**：编译出的二进制文件失去 “可移植性”—— 只能在和你当前 CPU 指令集兼容的机器上运行，放到指令集更老的 CPU 上会直接崩溃（比如在支持 AVX2 的 CPU 上编译，放到仅支持 SSE2 的老 CPU 上运行）。
- **适用场景**：仅在本机运行的程序（如个人工具、服务器程序，且服务器 CPU 固定）；如果需要跨机器部署，不要加这个参数（或指定通用 CPU 型号，如`-Ctarget-cpu=skylake`）。

#### 2. `-Cforce-frame-pointers=yes`

- **核心含义**：强制编译器生成 “栈帧指针（Frame Pointer）”，即使在优化模式下也不省略。
- **先理解栈帧指针**：
    
    栈帧指针（通常是 x86_64 的`rbp`寄存器）是用来标记当前函数调用栈起始位置的指针，调试器 / 性能分析工具（如`perf`、`gdb`）依赖它来解析函数调用栈（比如看程序卡在哪个函数、性能消耗在哪个调用链）。
- **默认行为**：
    
    在`release`模式（`opt-level ≥ 2`）下，Rust 编译器会省略栈帧指针（把`rbp`寄存器用作通用寄存器），以提升运行效率；但这会导致`perf`等工具无法准确解析调用栈（显示`[unknown]`）。
- **实际影响**：
    
    ✅ **优点**：能使用`perf`、`gprof`等性能分析工具精准分析`release`模式下的程序调用栈，定位性能瓶颈；
    
    ❌ **缺点**：极其轻微的性能损耗（因为少了一个通用寄存器可用），对绝大多数程序来说几乎感知不到（仅在极致性能要求的场景可能有影响）。
- **适用场景**：需要对`release`版本做性能剖析（Profiling）、定位性能问题时；如果不需要性能分析，可省略该参数以追求极致性能。

- `-Ctarget-cpu=native`：**提升本机运行性能，但牺牲二进制文件的可移植性**；
- `-Cforce-frame-pointers=yes`：**几乎无性能损耗，却能让性能分析工具正常工作**；


**`-Zbuild-std`**：
- 是 Rust 的**不稳定特性（nightly 版专属）**，`-Z`前缀表示启用 nightly 版的实验性功能；
- `build-std`的作用是**让用户手动编译 Rust 的标准库（std）**，而非使用 rustup 预编译好的标准库。

`-Zbuild-std`的典型使用场景：

- 需对标准库进行定制编译（比如启用标准库的特定优化、修改标准库代码）；
- 需交叉编译（比如编译为嵌入式平台的目标架构，而 rustup 未提供对应预编译的标准库）；
- 需将标准库与项目代码一起做全局优化（比如配合`lto = "fat"`实现更彻底的链接时优化）。

这条命令的完整效果是：
以**nightly 版 Rust**，在**发布模式**下编译项目，同时**手动编译 Rust 标准库**，并统计整个编译 + 运行过程的耗时。

### 注意事项

- `-Zbuild-std`是**nightly 版专属特性**，稳定版（stable）Rust 不支持该参数；
- 使用`-Zbuild-std`时，通常需要搭配`--target`指定目标架构（比如交叉编译时），否则默认编译为本机架构；
- 手动编译标准库会**显著增加编译时间**（因为需要额外编译整个标准库），这也是用`time`统计耗时的原因之一。

不使用`-Zbuild-std`的影响

`-Zbuild-std`的核心是 “手动编译 Rust 标准库”，不使用它时：

- Rust 编译器会直接使用`rustup`预编译好的**通用版标准库**（针对通用 CPU 架构、默认编译参数）；
- 你的项目代码会按`rustflags`配置编译，但标准库仍保持预编译的 “通用状态”，不会继承你的`rustflags`优化。
##### 查看性能
![](asserts/Pasted%20image%2020260106161030.png)

##### paru -Ss libc


这是一段在 Arch Linux（或基于 Arch 的发行版）终端中执行的命令，用于安装软件包，以下是解析：

###### 命令解析

1. **可见部分**：`paru -Ss libc`
    
    - `paru`：是 Arch Linux 的 AUR（用户软件仓库）助手，用于安装官方仓库及 AUR 的软件包；
    - `-Ss`：是`paru`的参数组合，`-S`表示 “同步并安装软件包”，`-s`表示 “搜索软件包”；
    - `libc`：是要搜索 / 安装的软件包名称（C 语言标准库，系统基础依赖）。
    
2. **模糊部分**：开头的 “e in userspace”
    
    - 推测是命令执行前的上下文信息（比如之前的命令输出或终端提示），不影响当前命令的核心含义。

###### 命令作用

`paru -Ss libc`的实际效果是：**搜索 Arch Linux 官方仓库及 AUR 中名称包含`libc`的软件包**，并列出搜索结果（包括包名、描述、仓库来源）。

这是一条在 Linux 终端中执行的命令，用于**分析 Rust 项目发布版本的性能（采集函数调用栈并生成性能报告）**，以下是逐部分解析：

### 命令拆解

1. **`perf record`**：
    
    - `perf`是 Linux 内核自带的性能分析工具；
    - `record`子命令的作用是 “采集程序运行时的性能数据”（包括 CPU 使用率、函数调用栈、事件计数等）。
    
2. **`--call-graph dwarf`**：
    
    - `--call-graph`：指定采集**函数调用栈**（用于定位性能瓶颈在哪个函数调用链）；
    - `dwarf`：表示通过**DWARF 调试信息**来解析函数调用栈（需要程序编译时生成调试信息，对应 Rust 中`[profile.release]`下的`debug = true`配置）。
    
3. **`cargo run --release`**：
    
    - `cargo run`：编译并运行 Rust 项目；
    - `--release`：以发布模式编译（启用最高级优化，生成性能最优的二进制文件）。
    

### 核心作用

这条命令的完整效果是：

以**发布模式**运行 Rust 项目，同时用`perf`采集程序运行时的性能数据（包含完整的函数调用栈），最终生成`perf.data`文件（性能数据记录）。

### 关键前提条件

要让这条命令正常工作，需满足：

1. Rust 项目的`[profile.release]`配置中开启了`debug = true`（否则`dwarf`无法解析调用栈）；
2. 系统已安装`perf`工具（Debian/Ubuntu：`sudo apt install linux-tools-common`；Arch：`sudo pacman -S perf`）；
3. 若采集到内核函数调用栈，需配置`/proc/sys/kernel/perf_event_paranoid`（临时设置：`sudo sysctl -w kernel.perf_event_paranoid=-1`）。

### 后续步骤（生成性能报告）

采集完成后，可通过以下命令分析`perf.data`：

bash

运行

```
# 生成文本格式的性能报告（按函数耗时排序）
perf report

# 生成火焰图（更直观的调用栈耗时可视化，需安装火焰图工具）
perf script | stackcollapse-perf.pl | flamegraph.pl > flamegraph.svg
```

### 总结

1. 这条命令是**Rust 发布版本的性能剖析工作流**，用于定位程序的性能瓶颈；
2. `--call-graph dwarf`依赖调试信息（`debug = true`），否则无法解析函数调用栈；
3. 采集后通过`perf report`或火焰图分析数据，可明确哪些函数占用了最多 CPU 资源。

![](asserts/Pasted%20image%2020260106162611.png)


它向你展示自己，也就是该函数的代价

A的自成本为执行A所花费的时间减去执行B所花费的时间。

执行`perf report`后，会进入交互式界面，核心关注以下内容：

1. **函数列表（按 CPU 占比排序）**
    
    - 顶部是 CPU 耗时占比最高的函数（比如某函数占比 20%，表示它消耗了 20% 的 CPU 时间）；
    - 函数名前的`[.]`表示用户态函数，`[k]`表示内核态函数。
    
2. **展开函数调用栈**
    
    - 选中某个函数，按`Enter`键可展开其**调用栈**（即 “谁调用了这个函数”）；
    - 示例：
        
        plaintext
        
        ```
        20.00%  brc  libbrc.so  [.] my_heavy_function
           20.00%  0x55f8a7b3c120
             20.00%  main  [.] main
        ```
        
        表示`main`调用了匿名函数`0x55f8a7b3c120`，最终调用`my_heavy_function`，该函数消耗了 20% 的 CPU。

![](asserts/Pasted%20image%2020260106162930.png)

![](asserts/Pasted%20image%2020260106163455.png)



```rust
________________________________________________________
Executed in   11.35 secs    fish           external
   usr time    9.90 secs    0.00 micros    9.90 secs
   sys time    0.28 secs  747.00 micros    0.28 secs


```

BtreeMap->HashMap
将entry拆分减少string分配
```rust
________________________________________________________
Executed in    6.58 secs    fish           external
   usr time    5.31 secs    0.00 millis    5.31 secs
   sys time    0.20 secs    1.05 millis    0.20 secs

```
不使用lines，直接使用字节，最后转换
```rust
________________________________________________________
Executed in    6.53 secs    fish           external
   usr time    5.40 secs  701.00 micros    5.40 secs
   sys time    0.22 secs    0.00 micros    0.22 secs
```

这里开始使用新的基准，基准是5亿数据
在使用从字节转换成String不check，速度提升了（5/60）
### 新基准
```
________________________________________________________
Executed in   37.71 secs    fish           external
   usr time   35.98 secs  208.00 micros   35.98 secs
   sys time    1.64 secs  375.00 micros    1.64 secs
```

### 优化：

```rust
    let f = std::fs::File::open("measurements3.txt").unwrap();
    let reader = BufReader::new(f);
```

这是一段一段读的，能不能让整个文件在内存里，一次读取

### 尝试使用内存映射，但是规则不准用外部库，不用
- [Rust-crate-memmap-基本概念](../../../../Areas/Rust/Area/3%20库/库/crate/memmap/Rust-crate-memmap-基本概念.md)
```
________________________________________________________
Executed in   33.17 secs    fish           external
   usr time   32.34 secs  786.00 micros   32.34 secs
   sys time    0.44 secs    0.00 micros    0.44 secs

```

那我们尝试看下memmap的源码

- [libc-mmap-基本概念](../4%20note/note/inbox/libc-mmap-基本概念.md)
 - [Rust-memmmap-map源码](../4%20note/note/inbox/Rust-memmmap-map源码.md)
- https://docs.rs/memmap2/latest/memmap2/enum.Advice.html
- https://man7.org/linux/man-pages/man2/madvise.2.html

```
 libc::madvise(ptr, len, libc::MADV_SEQUENTIAL);
```
- https://github.com/RazrFalcon/memmap2-rs/blob/a36f67f7649c16c390f11e0a4278f4d80fef8e9a/src/unix.rs#L424


plit(|c| c == &b'\n') 这个和 plit(|c| *c == b'\n') 性能有差距吗
- 对于 `u8`（一个 `Copy` 类型），Rust 会执行**按值复制**（copy），这是轻量级的 CPU 寄存器操作，不涉及堆或额外开销
对于 `Copy` 类型（整数、布尔、浮点、指针等），解引用不会有额外开销，它只是拷贝值到寄存器。

如果你知道你顺序读取东西，所以你不必增长，你不必copy。在程序运行时从后台读取，io并行性。

```rust
________________________________________________________
Executed in   33.95 secs    fish           external
   usr time   28.34 secs  262.00 micros   28.34 secs
   sys time    3.42 secs  361.00 micros    3.42 secs
```

![](asserts/Pasted%20image%2020260106213033.png)

修改cargo 配置不再丢失堆栈i信息
```toml

[build]

rustflags = ["-Ctarget-cpu=native", "-Cforce-frame-pointers=yes"]
```

![](asserts/Pasted%20image%2020260106213216.png)

- [Rust-cargo-Cforce-frame-pointers=yes](../4%20note/note/inbox/Rust-cargo-Cforce-frame-pointers=yes.md)

- 栈指针不是默认开启的。
### 修改为引用而不是分配字符串
```rust
Executed in   29.58 secs    fish           external
   usr time   28.38 secs    0.00 micros   28.38 secs
   sys time    0.61 secs  577.00 micros    0.61 secs
```
// `使用&[u8]只是可能，因为我们打破了顺序MADV_SEQUENTIAL`

### 解析浮点数花了10
![](asserts/Pasted%20image%2020260106220041.png)
基准时间
```
Executed in   29.82 secs    fish           external
   usr time   25.91 secs  411.00 micros   25.91 secs
   sys time    0.58 secs  437.00 micros    0.58 secs
```
修改后
```
________________________________________________________
Executed in   28.25 secs    fish           external
   usr time   24.75 secs    0.00 micros   24.75 secs
   sys time    0.46 secs  972.00 micros    0.46 secs
```


![](asserts/Pasted%20image%2020260109205811.png)

这是 **CPU 缓存 / 页表（TLB）的性能统计数据**（通常由 `perf stat` 等工具输出），记录了程序运行时的缓存 / 页表访问、命中 / 缺失情况，核心是反映内存访问的效率。

### 各指标含义

这些指标分为 **缓存（Cache）** 和 **页表（TLB）** 两类，以 “`-loads`（访问次数）” 和 “`-load-misses`（缺失次数）” 为核心：

#### 1. L1 数据缓存（L1-dcache）

- `L1-dcache-loads:u`：L1 数据缓存的**总访问次数**（466.47 亿次），速率 3.328 G / 秒。
- `L1-dcache-load-misses:u`：L1 数据缓存的**缺失次数**（51.63 亿次），缺失率 `1.10%`（占总访问的 1.10%）。
    
    → 说明 L1 数据缓存命中率很高（98.9%），内存访问效率不错。

#### 2. L1 指令缓存（L1-icache）

- `L1-icache-loads:u`：L1 指令缓存的**总访问次数**（2223.62 万次），速率 1.579 M / 秒。
- `L1-icache-load-misses:u`：L1 指令缓存的**缺失次数**（18.10 万次），缺失率 `0.81%`。
    
    → 指令缓存命中率也很高，指令读取效率好。

#### 3. 数据页表（dTLB）

TLB 是 “页表缓存”，用于加速虚拟地址到物理地址的转换；`dTLB` 是数据页的 TLB。

- `dTLB-loads:u`：dTLB 的**总访问次数**（307.75 万次），速率 218.548 K / 秒。
- `dTLB-load-misses:u`：dTLB 的**缺失次数**（249.35 万次），缺失率 `79.27%`。
    
    → dTLB 缺失率略高，但还在合理范围。

#### 4. 指令页表（iTLB）

`iTLB` 是指令页的 TLB。

- `iTLB-loads:u`：iTLB 的**总访问次数**（4.97 万次），速率 3.525 K / 秒。
- `iTLB-load-misses:u`：iTLB 的**缺失次数**（11.10 万次），缺失率 `223.52%`。
    
    → **这里有问题**：缺失次数超过总访问次数，通常是因为程序存在大量 “指令页切换”（比如代码段分散、频繁跳转），导致 iTLB 频繁失效，会拖慢程序执行（需要从内存 / 硬盘加载页表，延迟很高）。

### 核心结论

程序的 **L1 缓存效率很高**，但 **iTLB（指令页表缓存）缺失率异常高（223.52%）**，这是主要的性能瓶颈 —— 需要优化指令的局部性（比如减少代码段分散、避免频繁跳转）。
### 取消内联
将浮点数解析提取出函数，然后永不内联，可以看到函数内部的具体分析。

### 对于解析函数

消除循环

### split
![](asserts/Pasted%20image%2020260109220423.png)
性能损耗，逐个字符遍历并使用闭包来比较他们

memchr

`memchr` 通常是用**汇编语言**高度优化的，或者利用了 CPU 的 **SIMD（单指令多数据流）** 技术。

- [Rust-offset_from-地址偏移计算](../4%20note/note/inbox/Rust-offset_from-地址偏移计算.md)

两次取最低
之前
```
________________________________________________________
Executed in   23.48 secs    fish           external
   usr time   20.48 secs  880.00 micros   20.47 secs
   sys time    0.64 secs    0.00 micros    0.64 sec
```
之后
```
________________________________________________________
Executed in   23.20 secs    fish           external
   usr time   20.03 secs    1.06 millis   20.03 secs
   sys time    0.65 secs    0.00 millis    0.65 secs


```
![](asserts/Pasted%20image%2020260109231643.png)
### 使用simd替代查找分号（负优化？）
![](asserts/Pasted%20image%2020260110091543.png)
![](asserts/Pasted%20image%2020260110104335.png)

可能还变慢了
创建两个相同的方法比较时间

![](asserts/Pasted%20image%2020260110104645.png)
compare是优化前的方法

### 优化hashmap

![](asserts/Pasted%20image%2020260110105049.png)

hash里面makehash和get_mut占了所有时间的百分之60
看一下makehash里面有什么
![](asserts/Pasted%20image%2020260110105145.png)

默认哈希，对抗哈希冲突，秘钥生成过程中注入随机数据

如何对抗，使用btreemap避免hash，但是查找会慢很多
以下是数据
![](asserts/Pasted%20image%2020260110110910.png)
- 时间来到了1分钟
- 而查询来到了88%
![](asserts/Pasted%20image%2020260110122942.png)
**在使用ahasher后速度下降明显，显然哈希冲突是代价很高的动作。**

![](asserts/Pasted%20image%2020260110133325.png)



## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
