# Rcore 第四章 地址空间

> 写作就是思考，因为对于Rcore第四章整个知识脉络有点过于庞大，使用写作的方式重新梳理

[TOC]



## 引言

物理内存是操作系统需要管理的一个重要资源，无论是代码还是数据，程序的运行离不开内存。为了更好管理内存，提高内存的动态使用效率，通过隔离保证安全性，把有限内存变成无限内存。

### 现代操作系统实现什么功能

- 通过**动态内存分配**，提高了应用程序对内存的动态使用效率
  - how：
    - 页表机制
    - 算法
    - 缓存：TLB
  - why：
    - 应用执行结束，应用内存空间被释放，但是这块空间还驻留在内存里，其他应用无法利用
    - 应用自己分配不一定合理，比如应用80%代码可能不会执行，执行的就20%，这部分代码可以不用放在内存里
    - 内部内存碎片，外部内存碎片
- 通过页表的虚实内存映射机制
  - 简化：简化了编译器对应用的地址空间设置
    - how：使用抽象、透明易用的接口，应用开发者不需要考虑实际物理内存布局，地址可以统一使用内存布局
  - 安全：加强了应用之间，应用与内核之间的**内存隔离**，增强了系统安全
    - how：硬件和读写执行标志位和硬件机制，界限寄存器实现内存隔离，程序只能访问属于自己的内存
  - 通过页表的虚实内存映射机制，可以实现空分复用（提出，但没有实现）

### 问题：直接将程序放到物理内存，应用直接访问物理内存会产生什么问题？

- 不够透明，易用。（应用开发者视角）
  - 需要清楚计算机的物理内存布局以及自己被加载到哪个地址运行。
  - 为了避免冲突可能还需要应用的开发者们对此进行协商
- 没有任务保护和隔离
  - 可以读写其他应用的数据来窃取信息或者破坏其它应用的正常运行
  - 应用甚至可以直接来替换掉原本的 `trap_handler` 函数
- 内存使用空间在其运行前已经限定死了
  - 内核不能灵活地给应用程序提供的运行时动态可用内存空间。
  - 内存利用效率低

### 本章内容总结和导读

- TaskManager 的管理范围
  - 每个 Task 的上下文 Task Context 还包括该任务的地址空间
- 切换任务
  - 需要切换任务的地址空间

- 内核
  - 页帧分配、堆分配
  - 表示应用地址空间的 Apps MemSets 类型和内核自身地址空间的 Kernel MemSet`类型

- 应用程序视角

  - 不用考虑内存布局，`linker.ld`中应用程序的起始地址都改为 `0x10000`
  - why
    - 因为现在操作系统可将不同的应用虚拟地址映射到不同内存地址，开发者不用自己硬编码程序的物理地址了

- 内核

  - 增加连续内存分配

    - 代码：`os/src/mm/heap_allocator.rs`

  - 建立页表

    1. 首先，需要管理整个系统的物理内存
       - 这就需要知道整个计算机系统的物理内存空间的范围，物理内存中哪些区域是空闲可用的，哪些区域放置内核/应用的代码和数据。

    2. 以物理页帧为单位分配和回收物理内存
       - 代码：`os/src/mm/frame_allocator.rs`
    3. 虚拟内存中以各种粒度大小来动态分配内存资源
       - 代码：`os/src/mm/heap_allocator.rs`
    4. 地址转换函数
       - 虚拟地址、物理地址、虚拟页号、物理页号之间进行各种转换
       - 代码：os/src/mm/address.rs
    5. 建立页表
       - 页表项的数据结构表示，以及多级页表的起始物理页帧位置和整个所占用的物理页帧的记录
       - 代码：os/src/mm/page_table.rs
    6. 内核的恒等映射
       - 一旦使能分页机制，CPU 访问到的地址都是虚拟地址了，那么内核中也将基于虚地址进行虚存访问。所以在给应用添加虚拟地址空间前，内核自己也会建立一个页表，把整块物理内存通过简单的恒等映射（即虚拟地址映射到对等的物理地址）映射到内核虚拟地址空间中。
    7. 应用的相关数据结构
       - 这意味着第三章的初级 `task` 将进化到第四章的拥有独立页表的 `task`
    8. 虚拟地址空间的数据机构
       - MemorySet：地址空间这个抽象概念所对应的具象体现。
       - MapArea：在一个虚拟地址空间中，有代码段，数据段等不同属性且不一定连续的子空间，它们通过一个重要的数据结构 `MapArea` 来表示和管理。
       - 代码：os/src/mm/memory_set.rs
    9. 应用的ELF文件解析
       - 获取ELF执行文件数据：os/src/loader.rs
       - 解析 ELF 来创建一个应用地址空间：os/src/mm/memory_set
    10. 任务控制块
        - 任务控制块 `TaskControlBlock` 的管理范围，支持页表和单一虚拟地址空间
        - 代码：os/src/task/task.rs

    - 挑战
      - **程序运行的任务和管理应用的操作系统各自有独立的页表和虚拟地址空间**
        - **页表切换**
          - 上下文切换需要切换页表
          - 代码：os/src/trap/trap.S
        - 中断的分别处理
          - 需要对来自用户态和内核态的异常/中断分别进行处理
          - 代码：os/src/trap/mod.rs
      - **查页表以访问不同地址空间的数据**
        - 问应用地址空间中的数据跨了多个页，还需要注意处理地址的边界条件。
        - `os/src/syscall/fs.rs` `os/src/mm/page_table.rs ` `translated_byte_buffer`

### Rust 中的动态内存分配

### 静态与动态内存分配

#### 静态分配

- what
  - 基本定义：在程序**编译时**就**确定并分配内存空间**的方式，
    - 在代码中声明变量类型
    - 通过类型，编译器编译程序时已经知道这些变量所占的字节大小，于是给它们分配一块固定的内存将它们存储其中，这样变量在栈帧/数据段中的位置就被固定了下来
    - 其分配的内存**大小和位置**在程序运行期间是固定的。
- 固定的局限
  - 读文件的缓冲区，文件小，缓冲区过大，浪费，文件大，缓冲区小，就无法正常运行

#### 动态分配

**动态分配**就是随着应用的运行动态增减的内存空间 – 堆（Heap）。

应用管理堆：

- what
  - 在运行的时候从里面分配一块空间来存放变量，而在变量的生命周期结束之后，这块空间需要被回收以待后面的使用
- how
  - 应用所依赖的基础系统库（如 Linux 中的 glibc 库等）会直接通过系统调用（如类 Unix 内核提供的 `sbrk` 系统调用）来向内核请求增加/缩减应用地址空间内堆的大小，之后应用就可以基于基础系统库提供的内存分配/释放函数来获取和释放内存了。

- 内存碎片
  - 应用进行多次不同大小的内存分配和释放操作后，会产生内存空间的浪费，即存在无法被应用使用的空闲内存碎片。
  - 分类
    - 内碎片
      - 已被分配出去（属于某个在运行的应用）内存区域，占有这些区域的应用并不使用这块区域，操作系统也无法利用这块区域。
    - 外碎片
      - 已被分配出去（属于某个在运行的应用）内存区域，占有这些区域的应用并不使用这块区域，操作系统也无法利用这块区域。

- C语言标准库的动态分配的接口函数

  ```c
  void* malloc (size_t size);
  void free (void* ptr);
  ```

  - `malloc`：从堆中分配一块大小为 `size` 字节的空间，并返回一个指向它的指针。
  - `free`：在堆中回收这块空间

- 关于访问数据

  - 我们通过返回的指针变量来 **间接** 访问堆空间上的数据。
    - 事实上，我们在程序中能够 **直接** 看到的变量都是被静态分配在栈或者全局数据段上的，它们大小在编译期已知。比如我们可以把固定大小的指针放到栈（局部变量）或数据段（全局变量）上，然后通过指针来指向运行时才确定的堆空间上的数据，并进行访问。这样就可以通过确定大小的指针来实现对编译时大小不确定的堆数据的访问。

- 关于动态分配带来的额外功能：灵活调整变量的生命周期。

  - 一个局部变量被静态分配在它所在函数的栈帧中，一旦函数返回，这个局部变量的生命周期也就结束了；而静态分配在数据段中的全局变量则是在应用的整个运行期间均存在。
  - 动态分配允许我们构造另一种并不一直存在也**不绑定于函数调用的变量生命周期**：以 C 语言为例，可以说自 `malloc` 拿到指向一个变量的指针到 `free` 将它回收之前的这段时间，这个变量在堆上存在。由于需要跨越函数调用，我们需要作为堆上数据代表的变量在函数间以参数或返回值的形式进行传递，而这些变量一般都很小（如一个指针），其拷贝开销可以忽略。

- 动态分配的缺点

  - 它背后运行着连续内存分配算法，相比静态分配会带来一些额外的开销。如果动态分配非常频繁，可能会产生很多无法使用的内存碎片，甚至可能会成为应用的性能瓶颈。

## Rust 中的堆数据结构

- 在 Rust 中，与动态内存分配相关的智能指针主要有如下这些：

  - Box<T>：在创建时会在堆上分配一个类型为 `T` 的变量，它自身也只保存在堆上的那个变量的位置。
  - Rc<T>：单线程上使用的引用计数类型
  - Arc<Y>：多线程上使用的引用计数类型
  - `RefCell<T>` ：与 `Box<T>` 等智能指针不同，其 **借用检查** 在运行时进行。
  - `Mutex<T>` 是一个互斥锁，在多线程中使用。
    - 多线程同时访问，多个线程都拿到指向同一块堆上数据的 `Mutex<T>`
      - `Arc<Mutex<T>>` 
      - `Mutex<T>` 作为全局变量被分配到数据段

- 裸指针

  ![../_images/rust-containers.png](https://raw.githubusercontent.com/Levio-z/MyPicture/img/img/202504282211904.png)

基于上述智能指针，可形成更强大的 **集合** (Collection) 或称 **容器** (Container) 类型

- 向量 `Vec<T>` 类似于 C++ 中的 `std::vector` ；
- 键值对容器 `BTreeMap<K, V>` 类似于 C++ 中的 `std::map` ；
- 有序集合 `BTreeSet<T>` 类似于 C++ 中的 `std::set` ；
- 链表 `LinkedList<T>` 类似于 C++ 中的 `std::list` ；
- 双端队列 `VecDeque<T>` 类似于 C++ 中的 `std::deque` 。
- 变长字符串 `String` 类似于 C++ 中的 `std::string` 。

### 在内核中支持动态内存分配

相对于 C 语言而言，Rust语言在 `alloc` crate 中设定了一套简洁规范的接口，**只要实现了这套接口，内核就可以很方便地支持动态内存分配了**。Rust中智能指针或容器的底层实现，确实是通过 `alloc` crate 来完成的。`alloc` crate 提供了对堆内存的低级访问和管理能力。

- 标准库实现了这个create，因为标准库底层实现直接调用具体的平台的系统调用。当我们使用 Rust 标准库 `std` 的时候可以不用关心这个 crate ，因为标准库内已经已经实现了一套堆管理算法，并将 `alloc` 的内容包含在 `std` 名字空间之下让开发者可以直接使用。
- 然而操作系统内核运行在禁用标准库（即 `no_std` ）的裸机平台上，核心库 `core` 也并没有动态内存分配的功能，这个时候就要考虑利用 `alloc` 库定义的接口来实现基本的动态内存分配器。

#### 提供给它一个 `全局的动态内存分配器`来支持堆上的对象的动态分配

`alloc` 库需要我们提供给它一个 `全局的动态内存分配器` ，它会利用该分配器来管理堆空间，从而使得与堆相关的智能指针或容器数据结构可以正常工作。具体而言，我们的动态内存分配器需要实现它提供的 `GlobalAlloc` Trait，这个 Trait 有两个必须实现的抽象接口

```rust
// alloc::alloc::GlobalAlloc

pub unsafe fn alloc(&self, layout: Layout) -> *mut u8;
pub unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout);
```

- 可以看到，它们类似 C 语言中的 malloc/free ，分别代表堆空间的分配和回收，也同样使用一个裸指针（也就是地址）作为分配的返回值和回收的参数。

-  alloc::alloc::Layout 类型的参数
  - 它指出了分配的需求，分为两部分，分别是所需空间的大小 size ，以及返回地址的对齐要求 align 。这个对齐要求必须是一个 2 的幂次，单位为字节数，限制返回的地址必须是 align 的倍数。

> 而在 Rust 中，某些分配的对齐要求更灵活，有些的值可能很大，就只能采用更加复杂的方法。

#### 怎么实现呢？

- 然后只需将我们的动态内存分配器类型实例化为一个全局变量，并使用 `#[global_allocator]` 语义项标记即可。

由于该分配器的实现比较复杂，我们这里直接使用一个已有的伙伴分配器实现。首先添加 crate 依赖：

```rust
# os/Cargo.toml

buddy_system_allocator = "0.6"
```

接着，需要引入 `alloc` 库的依赖，由于它算是 Rust 内置的 crate ，我们并不是在 `Cargo.toml` 中进行引入，而是在 `main.rs` 中声明即可

```rust
// os/src/main.rs

extern crate alloc;
```

然后，根据 `alloc` 留好的接口提供全局动态内存分配器：

```rust
// os/src/mm/heap_allocator.rs
use crate::config::KERNEL_HEAP_SIZE;
// 使用buddy system（伙伴系统）作为分配算法的实现
use buddy_system_allocator::LockedHeap;

//  1.声明全局分配器，使用 alloc 要求的 #[global_allocator] 语义项进行标记。
#[global_allocator]
/// 2.实例化成一个全局变量的堆分配器实例，使用互斥锁包装以支持多线程访问
static HEAP_ALLOCATOR: LockedHeap = LockedHeap::empty();
/// 3. 使用静态可变数组作为堆内存空间，位于内核的 .bss 段中，包装在HEAP_ALLOCATOR
/// 堆空间，大小由 KERNEL_HEAP_SIZE 常量定义
static mut HEAP_SPACE: [u8; KERNEL_HEAP_SIZE] = [0; KERNEL_HEAP_SIZE];

/// 4.初始化堆分配器
/// 
/// 该函数需要在操作系统启动早期被调用，以便初始化内核堆分配器
/// 使用 unsafe 块是因为需要操作静态可变数组 HEAP_SPACE
pub fn init_heap() {
    unsafe {
        HEAP_ALLOCATOR
            .lock()
        	// 调用 init 方法告知它能够用来分配的空间的起始地址和大小即可。
            .init(HEAP_SPACE.as_ptr() as usize, KERNEL_HEAP_SIZE);
    }
}

/// 5.实现内存分配错误处理函数
#[alloc_error_handler]
/// 当堆内存分配发生错误时，打印错误信息并触发 panic
pub fn handle_alloc_error(layout: core::alloc::Layout) -> ! {
    panic!("Heap allocation error, layout = {:?}", layout);
}
/// 6.尝试一下动态内存分配吧！感兴趣的同学可以在 rust_main 中尝试调用下面的 heap_test 函数（调用 heap_test() 前要记得先调用 init_heap() ）。
/// 堆内存分配器的测试函数
/// 
/// 测试内容包括：
/// 	1. Box 智能指针的分配和释放
/// 	2. Vec 动态数组的创建、插入和遍历
/// 	3. 确保分配的内存位于 .bss 段内
#[allow(unused)]
pub fn heap_test() {
    use alloc::boxed::Box;
    use alloc::vec::Vec;
    // 声明外部符号，用于获取 .bss 段的起始和结束地址
    extern "C" {
        fn sbss();
        fn ebss();
    }
    // 计算 .bss 段的地址范围
    let bss_range = sbss as usize..ebss as usize;
    
    // 测试 1：Box 分配
    let a = Box::new(5);
    assert_eq!(*a, 5);
    // 验证分配的内存是否在 .bss 段内
    assert!(bss_range.contains(&(a.as_ref() as *const _ as usize)));
    drop(a);
    
    // 测试 2：Vec 动态数组
    let mut v: Vec<usize> = Vec::new();
    // 向 Vec 中插入数据
    for i in 0..500 {
        v.push(i);
    }
    // 验证数据正确性
    for (i, val) in v.iter().take(500).enumerate() {
        assert_eq!(*val, i);
    }
    // 验证 Vec 的内存分配是否在 .bss 段内
    assert!(bss_range.contains(&(v.as_ptr() as usize)));
    drop(v);
    println!("heap_test passed!");
}
```

## 地址空间

- 地址空间：
  - 由来：为了限制应用访问内存空间的范围并给操作系统提供内存管理的灵活性，计算机科学家提出了 **地址空间（Address Space）** 抽象，并在内核中建立虚实地址空间的映射机制，给应用程序提供一个基于地址空间的安全虚拟内存环境，让应用程序简单灵活地使用内存。
  - 实现过程：CPU 访问数据和指令的内存地址是虚地址，通过硬件机制（比如 MMU +页表查询）进行地址转换，找到对应的物理地址。

### 虚拟地址与地址空间

#### 地址虚拟化出现之前

- 裸机，整个硬件资源只用来执行单个裸机应用的时候，并不存在真正意义上的操作系统，而只能算是一种应用函数库。那个时候，物理内存的一部分用来保存函数库的代码和数据，余下的部分都交给应用来使用。，将应用占据的内存分成几个段：代码段、全局数据段、堆和栈等。当然，由于就只有这一个应用，它想如何调整布局都是它自己的事情。
- 批处理：和裸机应用很相似：批处理系统的每个应用也都是独占内核之外的全部内存空间，只不过当一个应用出错或退出之后，它所占据的内存区域会被清空，而应用序列中的下一个应用将自己的代码和数据放置进来。
- 多道程序：应用开始多出了一种“暂停”状态，这可能来源于它主动 yield 交出 CPU 资源，或是在执行了足够长时间之后被内核强制性放弃处理器。
  - 当应用处于暂停状态的时候，它驻留在内存中的代码、数据该何去何从呢？
    - 每个应用仍然和在批处理系统中一样独占内核之外的整块内存，当暂停的时候，内核负责将它的代码、数据保存在外存（如硬盘）中，然后把即将换入的应用在外存上的代码、数据恢复到内存，这些都做完之后才能开始执行新的应用。
      - 缺点：做法需要大量读写外部存储设备，而它们的速度都比 CPU 慢上几个数量级，这导致任务切换的开销过大
    - 限制每个应用的最大可用内存空间小于物理内存的容量，这样就可以同时把多个应用的数据驻留在内存中。
      - 很大程度上降低了任务切换的开销。
      - 缺点：
        - 多个应用驻留内存，内存资源使用效率低，每个应用只能占有固定的内存，分配不合理
- 从应用开发的角度看
  - 怎样实现：需要应用程序决定自己会被加载到哪个物理地址运行，需要直接访问真实的物理内存。
  - 副作用：这就要求应用开发者对于硬件的特性和使用方法有更多了解，产生额外的学习成本，也会为应用的开发和调试带来不便。

####  加一层抽象加强内存管理

抽象仍然是最重要的指导思想。在这里，抽象意味着内核要负责将**物理内存管理起来**，并**为上面的应用提供一层抽象接口**

- *透明* ：应用开发者可以不必了解底层真实物理内存的硬件细节，且在非必要时也不必关心内核的实现策略， **最小化他们的心智负担**；
- *高效* ：这层抽象至少在大多数情况下不应带来过大的额外开销；
- *安全* ：这层抽象应该有效检测并阻止应用读写其他应用或内核的代码、数据等一系列恶意行为。

**地址空间**：最终，到目前为止仍被操作系统内核广泛使用的抽象被称为 **地址空间** (Address Space) 。

- **应用程序的视角**
  - 操作系统分配给应用程序一个地址范围受限（容量很大），独占的连续地址空间（其中有些地方被操作系统限制不能访问，如内核本身占用的虚地址空间等），因此应用程序可以在划分给它的地址空间中随意规划内存布局，它的各个段也就可以分别放置在地址空间中它希望的位置（当然是操作系统允许应用访问的地址）。
  - 使用虚拟地址读写数据：
    - 应用同样可以**使用一个地址**作为索引来读写自己地址空间的数据，就像用物理地址作为索引来读写物理内存上的数据一样。
    - 应用能够直接看到并访问的内存就只有操作系统提供的地址空间，且它的**任何一次访存使用的地址都是虚拟地址**，无论取指令来执行还是读写栈、堆或是全局数据段都是如此。
  - 特点
    - 独占性：由于每个应用独占一个地址空间，里面只含有自己的各个段，于是它可以随意规划属于它自己的各个段的分布而**无需考虑和其他应用冲突**；
    - 安全性：**只能通过虚拟地址读写它自己的地址空间**，它完全无法窃取或者破坏其他应用的数据，毕竟那些段在其他应用的地址空间内，这是它没有能力去访问的。

![../_images/address-translation.png](https://raw.githubusercontent.com/Levio-z/MyPicture/img/img/202504290053966.png)

#### 增加硬件加速虚实地址转换

- CPU中的**内存管理单元** (MMU, Memory Management Unit)
  - 功能：自动将这个虚拟地址进行 **地址转换** (Address Translation) 变为一个物理地址
  - MMU实现虚实地址映射：在 MMU 的帮助下，应用对自己虚拟地址空间的读写才能被实际转化为对于物理内存的访问。
- 虚拟地址到物理地址的映射关系
  - 每个应用的地址空间都存在一个从虚拟地址到物理地址的映射关系。
  - 硬件通过寄存器区分不同映射关系
    - 不同的应用来说，该映射可能是不同的，要做到这一点，就需要硬件提供一些寄存器，软件可以对它进行设置来控制 MMU 按照哪个应用的地址映射关系进行地址转换。
- 软件部分的内核需要完成的重要工作：
  - 将应用的代码/数据放到物理内存并进行管理
  - 建立好应用的地址映射关系
  - 在任务切换时控制 MMU 选用应用的地址映射关系

### 地址空间只是一层抽象接口，它有很多种具体的实现策略。

下面我们简要介绍几种曾经被使用的策略，并探讨它们的优劣。

#### 分段内存管理

![../_images/simple-base-bound.png](https://raw.githubusercontent.com/Levio-z/MyPicture/img/img/202504290113659.png)



##### 划分为若干个大小相同的 **插槽** (Slot) 

- 固定大小
  - 地址空间大小限制为一个固定的常数 bound ，也即每个应用的可用虚拟地址区间均为 [0,bound]。

- 实现过程
  - 将物理内存除了内核预留空间之外的部分划分为若干个大小相同的 **插槽** (Slot)
  - 应用的物理空间：假设其起始物理地址为base，应用对应的实际物理地址[base,base+bound)
  - 地址转换：检查一下虚拟地址不超过地址空间的大小限制（此时需要借助特权级机制通过异常来进行处理），然后做一个线性映射，将虚拟地址加上base就得到了数据实际所在的物理地址。

- 硬件和内核的功能

  - 硬件：MMU 只需要base,bound两个寄存器，在地址转换进行比较或加法运算即可
  - 内核：任务切换时完成**切换base寄存器**。在对一个应用的内存管理方面，只需考虑一组插槽的占用状态，可以用一个 **位图** (Bitmap) 来表示，随着应用的新增和退出对应置位或清空。

- 问题

  - **地址空间预留了一部分：可能浪费的内存资源过多**。

  - **产生内碎片**：空间是固定的，需求较低的应用根本无法充分利用

    > **原因细节**：注意到应用地址空间预留了一部分，它是用来让栈得以向低地址增长，同时允许堆往高地址增长（支持应用运行时进行动态内存分配）。能力范围之内的消耗内存最多的应用分配，**需求较低的应用根本无法充分利用**每个应用的情况都不同，内核只能按照在它能力范围之内的消耗内存最多的应用的情况来统一指定地址空间的大小，而其他内存需求较低的应用根本无法充分利用内核给他们分配的这部分空间。但这部分空间又是一个完整的插槽的一部分，也不能再交给其他应用使用。
    >
    > **内碎片**：这种在已分配/使用的地址空间内部无法被充分利用的空间就是 **内碎片** (Internal Fragment) ，它限制了系统同时共存的应用数目。如果应用的需求足够多样化，那么内核无论如何设置应用地址空间的大小限制也不能得到满意的结果。这就是固定参数的弊端：虽然实现简单，但不够灵活。

![../_images/segmentation.png](https://raw.githubusercontent.com/Levio-z/MyPicture/img/img/202504290112499.png)

逻辑段：注意到内核开始以更细的粒度，也就是应用地址空间中的一个逻辑段作为单位来安排应用的数据在物理内存中的布局。对于每个段来说，不同的线性映射一对不同的 base/bound 进行区分，当任务切换的时候，这些对寄存器也需要被切换。

是否存在内存碎片问题？

- 按需分配，不再有内碎片了

  - 注意到每个段都只会在内存中占据一块与它实际所用到的大小相等的空间。

  - 堆的情况可能比较特殊，它的大小可能会在运行时增长，但是那需要应用通过系统调用向内核请求。

- 段大小不同

  - 段的大小都是不同的（它们可能来自不同的应用，功能也不同），内核更加通用、也更加复杂的连续内存分配算法来进行内存管理连续，内存分配算法就是每次需要分配一块连续内存来存放一个段的数据。随着一段时间的分配和回收，物理内存还剩下一些相互不连续的较小的可用连续块，其**中有一些只是两个已分配内存块之间的很小的间隙，它们自己可能由于空间较小，已经无法被用于分配**，这就是 **外碎片** (External Fragment) 。
  - 将这些不连续的外碎片“拼起来”，形成一个大的连续块。开销极大。

#### 分页内存管理

- 分页：结合固定插槽和分段的优点，按需分配的固定插槽，且插槽比较小
  - 分段大小不一产生外碎片的原因，在每个应用的地址空间大小均相同的情况下，只需利用类似位图的数据结构维护一组插槽的占用状态，从逻辑上分配和回收都是以一个固定的比特为单位，自然也就不会存在外碎片了。但是这样粒度过大，不够灵活，又在地址空间内部产生了内碎片。
  - 若要结合二者的优点的话，就需要内核始终以一个同样大小的单位来在物理内存上放置应用地址空间中的数据，这样内核就可以使用简单的插槽式内存管理，使得内存分配算法比较简单且不会产生外碎片；同时，这个单位的大小要足够小，从而其内部没有被用到的内碎片的大小也足够小，尽可能提高内存利用率。这便是我们将要介绍的分页内存管理。

![../_images/page-table.png](https://rcore-os.cn/rCore-Tutorial-Book-v3/_images/page-table.png)

- 内核以页为单位进行物理内存管理
  - 地址空间可以被分成若干个（虚拟） **页面** (Page) ，而可用的物理内存也同样可以被分成若干个（物理） **页帧** (Frame) ，虚拟页面和物理页帧的大小相同。
  - 应用地址空间：分页内存管理的粒度更小且大小固定，应用地址空间中的每个逻辑段都由多个虚拟页面组成。而且每个虚拟页面在地址转换的过程中都使用**与运行的应用绑定的不同的线性映射**，而不像分段内存管理那样每个逻辑段都使用一个相同的线性映射。

- 方便地址转换的编号

  - **我们给每个虚拟页面和物理页帧一个编号，虚拟页号** (VPN, Virtual Page Number) 和 **物理页号** (PPN, Physical Page Number) 

- 页表

  - 每个应用都有一个表示地址映射关系的 **页表** (Page Table) ，里面记录了该应用地址空间中的每个虚拟页面映射到物理内存中的哪个物理页帧，即数据实际被内核放在哪里。

  - 内容：

    - 记录地址映射关系
      - 其键的类型为虚拟页号，值的类型则为物理页号。当 MMU 进行地址转换的时候，虚拟地址会分为两部分（虚拟页号，页内偏移），MMU首先找到虚拟地址所在虚拟页面的页号，然后查当前应用的页表，根据虚拟页号找到物理页号；最后按照虚拟地址的页内偏移，给物理页号对应的物理页帧的起始地址加上一个偏移量，这就得到了实际访问的物理地址。

    - 置了一组保护位
      - `rwx` ， `r` 表示当前应用可以读该内存； `w` 表示当前应用可以写该内存； `x` 则表示当前应用可以从该内存取指令用来执行。
      - 一旦违反了这种限制则会触发异常，并被内核捕获到。通过适当的设置，可以检查一些应用在运行时的明显错误：比如应用修改只读的代码段，或者从数据段取指令来执行。

  - 存储：
    - 存在被内核管理的数据结构放在内存中，但是 CPU 也会直接访问它来查页表，这也就需要内核和硬件之间关于页表的内存布局达成一致。
    - 页表中的项数会很多，CPU 内也没有足够的硬件资源能够将它存下来

## SV39 多级页表的硬件机制

- 虚拟地址与物理地址的访问属性（可读，可写，可执行等），组成结构（页号，帧号，偏移量等），访问的空间范围，硬件实现地址转换的多级页表访问过程等；
- 如何用Rust语言来设计有类型的页表项。

### 虚拟地址和物理地址

#### 内存控制相关的CSR寄存器

- **启用MMU**
  - MMU默认不启用
    - 默认情况下 MMU 未被使能（启用），此时无论 CPU 位于哪个特权级，访存的地址都会作为一个物理地址交给对应的内存控制单元来直接访问物理内存。
  - 开启：修改 S 特权级的一个名为 `satp` 的 CSR 来启用分页模式
    - 在这之后 S 和 U 特权级的访存地址会被视为一个虚拟地址，它需要经过 MMU 的地址转换变为一个物理地址，再通过它来访问物理内存；而 M 特权级的访存地址，我们可设定是内存的物理地址。

![../_images/satp.png](https://rcore-os.cn/rCore-Tutorial-Book-v3/_images/satp.png)

- 上图是 RISC-V 64 架构下 `satp` 的字段分布，含义如下：
  - `MODE` 控制 CPU 使用哪种页表实现；
    -  0：所有访存都被视为物理地址
    - 8：SV39 分页机制被启用
      -  S/U 特权级的访存被视为一个 39 位的虚拟地址，它们需要先经过 MMU 的地址转换流程，如果顺利的话，则会变成一个 56 位的物理地址来访问物理内存；否则则会触发异常，这体现了分页机制的内存保护能力。
  - `ASID` 表示地址空间标识符，这里还没有涉及到进程的概念，我们不需要管这个地方；
  - PPN 存的是根页表所在的物理页号。

#### 地址格式与组成

![../_images/sv39-va-pa.png](https://rcore-os.cn/rCore-Tutorial-Book-v3/_images/sv39-va-pa.png)

- 单个页面的大小设置为4KiB，每个虚拟页面和物理页帧都对齐到这个页面大小
  - 4KiB需要用 12 位字节地址来表示，
- 虚拟地址和物理地址都被分成两部分
  - 它们的低 12 位，即[11:0]被称为**页内偏移**(Page Offset)，它描述一个地址指向的字节在它所在页面中的相对位置。
  - 而虚拟地址的高 27 位，即[38:12]为它的虚拟页号 VPN，同理物理地址的高 44 位，即[55:12]为它的物理页号 PPN，页号可以用来定位一个虚拟/物理地址属于哪一个虚拟页面/物理页帧。
- 地址转换过程：
  - 地址转换是以页为单位进行的，在地址转换的前后地址的页内偏移部分不变。可以认为 MMU 只是从虚拟地址中取出 27 位虚拟页号，在页表中查到其对应的物理页号（如果存在的话），最后将得到的44位的物理页号与虚拟地址的12位页内偏移依序拼接到一起就变成了56位的物理地址。

#### 地址相关的数据结构抽象与类型定义

- 将地址相关数据结构声明Rust类型元组结构体，可以看成usize的简单包装。抽象出不同的类型是为了方便且安全的进行类型转换来构建页表。

  - 物理地址

  - 虚拟地址

  - 物理页号

  - 虚拟页号

```rust
// os/src/mm/address.rs

#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub struct PhysAddr(pub usize);

#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub struct VirtAddr(pub usize);

#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub struct PhysPageNum(pub usize);

#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub struct VirtPageNum(pub usize);
```

- 实现From来实现类型与usize互相转换

  ```Rust
  // os/src/mm/address.rs
  
  const PA_WIDTH_SV39: usize = 56;
  const PPN_WIDTH_SV39: usize = PA_WIDTH_SV39 - PAGE_SIZE_BITS;
  // 从一个 usize 来生成 PhysAddr 即 PhysAddr::from(_: usize) 将得到一个 PhysAddr 
  impl From<usize> for PhysAddr {
      fn from(v: usize) -> Self { Self(v & ( (1 << PA_WIDTH_SV39) - 1 )) }
  }
  impl From<usize> for PhysPageNum {
      fn from(v: usize) -> Self { Self(v & ( (1 << PPN_WIDTH_SV39) - 1 )) }
  }
  
  impl From<PhysAddr> for usize {
      fn from(v: PhysAddr) -> Self { v.0 }
  }
  impl From<PhysPageNum> for usize {
      fn from(v: PhysPageNum) -> Self { v.0 }
  }
  ```

  - usize 来生成 `PhysAddr `、`VirtAddr`
    - 注意 SV39 支持的物理地址位宽为 56 位，因此在生成 `PhysAddr` 的时候我们仅使用 `usize` 较低的 56 位。
    - 同理在生成虚拟地址 的时候仅使用 `usize` 较低的 39 位。
  - 反过来，从类型生成对应的 `usize` 。
    - 其实由于我们在声明结构体的时候将字段公开了出来，从物理地址变量 `pa` 得到它的 usize 表示的更简便方法是直接 `pa.0` 。

> **Rust Tips：类型转换之 From 和 Into**
>
> 一般而言，当我们为类型 `U` 实现了 `From<T>` Trait 之后，可以使用 `U::from(_: T)` 来从一个 `T` 类型的实例来构造一个 `U` 类型的实例；而当我们为类型 `U` 实现了 `Into<T>` Trait 之后，对于一个 `U` 类型的实例 `u` ，可以使用 `u.into()` 来将其转化为一个类型为 `T` 的实例。

其次，地址和页号之间可以相互转换。我们这里仍以物理地址和物理页号之间的转换为例：
```rust
// os/src/mm/address.rs

impl PhysAddr {
    // 根据物理地址获取页内偏移
    pub fn page_offset(&self) -> usize { self.0 & (PAGE_SIZE - 1) }
}

impl From<PhysAddr> for PhysPageNum {
    // 从物理地址获取物理页号
    fn from(v: PhysAddr) -> Self {
        assert_eq!(v.page_offset(), 0);
        v.floor()
    }
}
// 从物理页号获取物理地址
impl From<PhysPageNum> for PhysAddr {
    fn from(v: PhysPageNum) -> Self { Self(v.0 << PAGE_SIZE_BITS) }
}
```

- 其中 `PAGE_SIZE` 为 4096 ， `PAGE_SIZE_BITS` 为 12 ，它们均定义在 `config` 子模块中，分别表示每个页面的大小和页内偏移的位宽。

- 对于不对齐的情况，物理地址不能通过 `From/Into` 转换为物理页号，而是需要通过它自己的 `floor` 或 `ceil` 方法来进行下取整或上取整的转换。

  ```rust
  // os/src/mm/address.rs
  
  impl PhysAddr {
      // PhysAddr/PAGE_SIZE = PhysPageNum
      pub fn floor(&self) -> PhysPageNum { PhysPageNum(self.0 / PAGE_SIZE) }
      pub fn ceil(&self) -> PhysPageNum { PhysPageNum((self.0 + PAGE_SIZE - 1) / PAGE_SIZE) }
  }
  ```

### 页表项的数据结构抽象与类型定义

![../_images/sv39-pte.png](https://rcore-os.cn/rCore-Tutorial-Book-v3/_images/sv39-pte.png)

 **页表项** (PTE, Page Table Entry)

- 物理页号：[53:10]是物理页号
- 最低的8位[7:0]是标志位
  - V(Valid)：仅当位 V 为 1 时，页表项才是合法的；
  - R(Read)/W(Write)/X(eXecute)：分别控制索引到这个页表项的对应虚拟页面是否允许读/写/执行；
  - U(User)：控制索引到这个页表项的对应虚拟页面是否在 CPU 处于 U 特权级的情况下是否被允许访问；
  - G：暂且不理会；
  - A(Accessed)：处理器记录自从页表项上的这一位被清零之后，页表项的对应虚拟页面是否被访问过；
  - D(Dirty)：处理器记录自从页表项上的这一位被清零之后，页表项的对应虚拟页面是否被修改过。

- 除了 `G` 外的上述位可以被操作系统设置，只有 `A` 位和 `D` 位会被处理器动态地直接设置为 `1` ，表示对应的页被访问过或修过（ 注：`A` 位和 `D` 位能否被处理器硬件直接修改，取决于处理器的具体实现）。让我们先来实现页表项中的标志位 `PTEFlags` ：

- ```rust
  // os/src/main.rs
  
  #[macro_use]
  extern crate bitflags;
  
  // os/src/mm/page_table.rs
  
  use bitflags::*;
  
  bitflags! {
      pub struct PTEFlags: u8 {
          const V = 1 << 0;
          const R = 1 << 1;
          const W = 1 << 2;
          const X = 1 << 3;
          const U = 1 << 4;
          const G = 1 << 5;
          const A = 1 << 6;
          const D = 1 << 7;
      }
  }
  ```

#### 页表项 `PageTableEntry` 

```rust
// 文件位置：os/src/mm/page_table.rs

/// 定义页表项结构体，每一项用来描述虚拟页到物理页的映射关系。
#[derive(Copy, Clone)]  // 支持按位拷贝（Copy）和显式克隆（Clone）
#[repr(C)]              // 按 C 语言标准布局内存，保证 bits 字段布局固定
pub struct PageTableEntry {
    pub bits: usize,    // 页表项的原始二进制表示（通常一个指针宽度）
}

impl PageTableEntry {
    /// 创建一个新的页表项，给定物理页号（PPN）和页表标志（Flags）
    pub fn new(ppn: PhysPageNum, flags: PTEFlags) -> Self {
        PageTableEntry {
            // 页号左移10位（保留低10位给flags），与flags按位或，组合成完整bits
            bits: ppn.0 << 10 | flags.bits as usize,
        }
    }

    /// 创建一个空页表项（bits=0，表示无效条目）
    pub fn empty() -> Self {
        PageTableEntry {
            bits: 0,
        }
    }

    /// 提取页表项中存储的物理页号（PPN）
    pub fn ppn(&self) -> PhysPageNum {
        // 右移10位去除flags部分，仅保留高位的物理页号
        // 再mask取出44位（RV64 Sv39 页表格式），确保有效位
        (self.bits >> 10 & ((1usize << 44) - 1)).into()
    }

    /// 提取页表项中存储的标志位（Flags）
    pub fn flags(&self) -> PTEFlags {
        // 截取bits低8位（u8），转换为PTEFlags结构
        PTEFlags::from_bits(self.bits as u8).unwrap()
    }
}

```

- 实现 `Copy/Clone` Trait
  - 来让这个类型以值语义赋值/传参的时候不会发生所有权转移，而是拷贝一份新的副本。
  - 从这一点来说 `PageTableEntry` 就和 usize 一样，因为它也只是后者的一层简单包装，并解释了 usize 各个比特段的含义。
- 内容
  - `PhysPageNum` 
  - 一个页表项标志位 `PTEFlags` 

- 后面我们还为 `PageTableEntry` 实现了一些辅助函数(Helper Function)，可以快速判断一个页表项的 `V/R/W/X` 标志位是否为 1 以 `V` 标志位的判断为例：

  ```rust
  // os/src/mm/page_table.rs
  
  impl PageTableEntry {
      pub fn is_valid(&self) -> bool {
          // PTEFlags::V看成是v的掩码，直接获取对应位上值
          (self.flags() & PTEFlags::V) != PTEFlags::empty()
      }
  }
  ```

### 多级页表

- 线性表
  - 页表的一种最简单的实现是线性表，也就是按照地址从低到高、输入的虚拟页号从 0 开始递增的顺序依次在内存中（我们之前提到过页表的容量过大无法保存在 CPU 中）放置每个虚拟页号对应的页表项。
  - 线性表的问题在于：它保存了所有虚拟页号对应的页表项，但是高达 512GiB 的地址空间中真正会被应用使用到的只是其中极小的一个子集

![../_images/linear-table.png](https://rcore-os.cn/rCore-Tutorial-Book-v3/_images/linear-table.png)

- 参见**地址格式与组成**

  - ![../_images/sv39-va-pa.png](https://rcore-os.cn/rCore-Tutorial-Book-v3/_images/sv39-va-pa.png)

  - 虚拟页号有2的27次方种，每个虚拟页都需要对应一个8字节的页表项

  - 将所有虚拟页号和物理页号的映射关系存下来，需要一共2的30次方字节，正好1G内存，占用过大，但是被应用使用到的只是其中极小的一个子集

- 使用**按需分配**优化

  - 只保存使用到的虚拟页号的页表项

- 非叶节点（页目录表，非末级页表）的表项标志位含义和叶节点（页表，末级页表）相比有一些不同：

  - 当 `V` 为 0 的时候，代表当前指针是一个空指针，无法走向下一级节点，即该页表项对应的虚拟地址范围是无效的；
  - 只有当 `V` 为1 且 `R/W/X` 均为 0 时，表示是一个合法的页目录表项，其包含的指针会指向下一级的页表；
  - 注意: 当 `V` 为1 且 `R/W/X` 不全为 0 时，表示是一个合法的页表项，其包含了虚地址对应的物理页号。

  ![../_images/pte-rwx.png](https://rcore-os.cn/rCore-Tutorial-Book-v3/_images/pte-rwx.png)

### SV39 地址转换过程

- 一个物理页帧能保存多少个页表项呢？
  - 页帧大小（12位）/页表项大小（3位）8字节 = 页帧保存的页表项个数结果为512（9位）
  - 地址以字节为最小分割
- 3级页表
  - 一个页只能放512个
- 一级页表，页目录表，通过**前9位**，定位到一个PTE，页表项存着二级页表的PPN
- 二级页表，页目录表，通过中间9位，定位到一个PTE，里面存着3级页表的PPN
- 三级页表，末级页表，通过最后9位，找到PTE，存着程序使用内存中页表的PPN
- ![../_images/sv39-full.png](https://rcore-os.cn/rCore-Tutorial-Book-v3/_images/sv39-full.png)

### TLB

- 实践表明绝大部分应用程序的虚拟地址访问过程具有时间局部性和空间局部性的特点。因此，在 CPU 内部，我们使用MMU中的 **快表（TLB, Translation Lookaside Buffer）** 来作为虚拟页号到物理页号的映射的页表缓存。

- MMU 应该如何知道当前做地址转换的时候要查哪一个页表呢？
  - PPN 字段指的就是多级页表根节点所在的物理页号
    - 因此，每个应用的地址空间就可以用包含了它多级页表根节点所在物理页号的 `satp` CSR 代表。在我们切换任务的时候， `satp` 也必须被同时切换。

- 什么情况需要刷新TLB

  - 但如果**修改了 satp 寄存器**，说明内核**切换到了一个与先前映射方式完全不同的页表**。此时快表里面存储的映射已经失效了，这种情况下内核要在修改 satp 的指令后面马上使用 sfence.vma 指令刷新清空整个 TLB。

  - 同样，我们**手动修改一个页表项**之后，也修改了映射，但 TLB 并不会自动刷新清空，我们也需要使用 sfence.vma 指令刷新整个 TLB。注：可以在 sfence.vma 指令后面加上一个虚拟地址，这样 sfence.vma 只会刷新TLB中关于这个虚拟地址的单个映射项。

## 管理 SV39 多级页表

### 物理页帧管理

当Bootloader把操作系统内核加载到物理内存中后，物理内存上已经有一部分用于放置内核的代码和数据。我们需要将剩下的空闲内存以单个物理页帧为单位管理起来，当需要存放应用数据或扩展应用的多级页表时分配空闲的物理页帧，并在应用出错或退出的时候回收应用占有的所有物理页帧。

#### 可用物理页的分配与回收

首先，我们需要知道物理内存的哪一部分是可用的。在 `os/src/linker.ld` 中，我们用符号 `ekernel` 指明了内核数据的终止物理地址，在它之后的物理内存都是可用的。而在 `config` 子模块中：

```rust
// os/src/config.rs

pub const MEMORY_END: usize = 0x80800000;
```

- 可用内存大小设置为 `8MiB `

  - 硬编码整块物理内存起始物理地址为 `0x80000000`,终止物理地址为 `0x80800000` 。 

  - 这个区间将被传给我们后面实现的物理页帧管理器用于初始化。

#### 物理页帧管理器需要提供哪些功能：

`FrameAllocator` Trait

```rust
// os/src/mm/frame_allocator.rs

trait FrameAllocator {
    fn new() -> Self;
    fn alloc(&mut self) -> Option<PhysPageNum>;
    fn dealloc(&mut self, ppn: PhysPageNum);
}
```

- 创建：即创建一个物理页帧管理器的实例
- 分配和回收：以及以物理页号为单位进行**物理页帧的分配和回收。**

#### 物理页帧管理器实现

我们实现一种最简单的栈式物理页帧管理策略 `StackFrameAllocator` ：

##### 首先，定义需要的结构体

```rust
// os/src/mm/frame_allocator.rs

pub struct StackFrameAllocator {
    current: usize,  //空闲内存的起始物理页号
    end: usize,      //空闲内存的结束物理页号
    recycled: Vec<usize>,
}
```

- 从未分配的区域：物理页号区间 [ `current` , `end` ) 此前均 *从未* 被分配出去过
- 回收的物理页号：向量 `recycled` 以后入先出的方式保存了被回收的物理页号

##### 实现FrameAllocator 

```rust
impl StackFrameAllocator {
    // 真正被使用起来之前，需要调用 init 方法
    pub fn init(&mut self, l: PhysPageNum, r: PhysPageNum) {
        self.current = l.0;
        self.end = r.0;
        // trace!("last {} Physical Frames.", self.end - self.current);
    }
}
// 实现FrameAllocator方法的三个功能
impl FrameAllocator for StackFrameAllocator {
    // new方法就是初始化，新增一个队列
    fn new() -> Self {
        Self {
            current: 0,
            end: 0,
            recycled: Vec::new(),
        }
    }
    // 分配方法
    fn alloc(&mut self) -> Option<PhysPageNum> {
        // 1. 栈 recycled 内有没有之前回收的物理页号，如果有的话直接弹出栈顶并返回
        if let Some(ppn) = self.recycled.pop() {
            Some(ppn.into())
        // 2. 内存耗尽分配失败的情况：即 recycled 为空且 current == end
        } else if self.current == self.end {
            None
        } else {
        // 3. 分配一个出去
            self.current += 1;
            // 使用之前实现的特征，调用into()从usize转换成PhysPageNum
            Some((self.current - 1).into())
        }
    }
    // 回收
    fn dealloc(&mut self, ppn: PhysPageNum) {
        let ppn = ppn.0;
        // validity check
        // 页面是否合法，存在已分配的区域，没有检测小于内核分配地址的区域，防止内核其他部分出错
        if ppn >= self.current || self.recycled.iter().any(|&v| v == ppn) {
            panic!("Frame ppn={:#x} has not been allocated!", ppn);
        }
        // recycle
        // 放到队列
        self.recycled.push(ppn);
    }
}
```

##### 创建全局实例

```rust
// os/src/mm/frame_allocator.rs

use crate::sync::UPSafeCell;
type FrameAllocatorImpl = StackFrameAllocator;
lazy_static! {
    pub static ref FRAME_ALLOCATOR: UPSafeCell<FrameAllocatorImpl> = unsafe {
        UPSafeCell::new(FrameAllocatorImpl::new())
    };
}
```

- 使用 Rust 的安全封装（如 `RefCell`）**避免显式 `unsafe`**。
  - 本结构用于单处理器系统中的**安全内部可变性（interior mutability）**；
  - 在需要可变访问时，调用公开的 `exclusive_access()` 方法即可。

##### 正式使用前需要初始化

```Rust
// os/src/mm/frame_allocator.rs

pub fn init_frame_allocator() {
    extern "C" {
        fn ekernel();
    }
    FRAME_ALLOCATOR
        .exclusive_access()
        // 
    	// 
        .init(PhysAddr::from(ekernel as usize).ceil(), PhysAddr::from(MEMORY_END).floor());
}
```

回顾下ekernel是什么？

- 区间的左端点应该是在`link.ld`定义的 `ekernel` 的物理地址以上取整方式转化成的物理页号，`ekernel` 表示内核映像（kernel image）在内存中的结束地址。
  - 开始位置向上取整是为了不破坏之前的内核的映像的数据
- 区间的右端点应该是 `MEMORY_END` 以下取整方式转化成的物理页号。
  - 结束位置向下取整是为了最后一个页帧不超过限制的界限。

```Rust
// os/src/mm/address.rs

impl PhysAddr {
    // PhysAddr/PAGE_SIZE = PhysPageNum
    pub fn floor(&self) -> PhysPageNum { PhysPageNum(self.0 / PAGE_SIZE) }
    pub fn ceil(&self) -> PhysPageNum { PhysPageNum((self.0 + PAGE_SIZE - 1) / PAGE_SIZE) }
}
```

#### 分配/回收物理页帧的接口

公开给其他内核模块调用的分配/回收物理页帧的接口：

```rust
// os/src/mm/frame_allocator.rs

pub fn frame_alloc() -> Option<FrameTracker> {
    FRAME_ALLOCATOR
        .exclusive_access()
        .alloc()
        .map(|ppn| FrameTracker::new(ppn))
}

fn frame_dealloc(ppn: PhysPageNum) {
    FRAME_ALLOCATOR
        .exclusive_access()
        .dealloc(ppn);
}
// os/src/mm/frame_allocator.rs
// RAII（资源获取即初始化),一旦变量离开作用域（析构），它自动释放资源。
impl Drop for FrameTracker {
    fn drop(&mut self) {
        frame_dealloc(self.ppn);
    }
}
```

- `frame_alloc` 的返回值类型并不是 `FrameAllocator` 要求的物理页号 `PhysPageNum` ，而是将其进一步包装为一个 `FrameTracker
- 这里借用了 RAII 的思想，将一个物理页帧的生命周期绑定到一个 `FrameTracker` 变量上，当一个 `FrameTracker` 被创建的时候，我们需要从 `FRAME_ALLOCATOR` 中分配一个物理页帧：

> **RAII（资源获取即初始化）** 是 C++ 中提出的一种资源管理机制，Rust 中被广泛继承：
>
> 一旦变量被创建（构造），它就持有某种资源；一旦变量离开作用域（析构），它自动释放资源。
>
> 在 Rust 中，RAII 通过 `Drop` trait 实现自动释放资源（类似析构函数），是所有安全内存管理的核心机制。

```
// os/src/mm/frame_allocator.rs

pub struct FrameTracker {
    pub ppn: PhysPageNum,
}

impl FrameTracker {
    pub fn new(ppn: PhysPageNum) -> Self {
        // page cleaning
        // 直接将这个物理页帧上的所有字节清零。
        // 由于这个物理页帧之前可能被分配过并用做其他用途,比如被其他应用使用
        let bytes_array = ppn.get_bytes_array();
        for i in bytes_array {
            *i = 0;
        }
        Self { ppn }
    }
}
```

最后做一个小结：从其他内核模块的视角看来，物理页帧分配的接口是调用 `frame_alloc` 函数得到一个 `FrameTracker` （如果物理内存还有剩余），它就代表了一个物理页帧，当它的生命周期结束之后它所控制的物理页帧将被自动回收。

### 多级页表管理

#### 页表基本数据结构与访问接口

我们知道，SV39 多级页表是以节点为单位进行管理的。每个节点恰好存储在一个物理页帧中，它的位置可以用一个物理页号来表示。

```Rust
// os/src/mm/page_table.rs

pub struct PageTable {
    // 保存三级页表的根节点物理页号
    root_ppn: PhysPageNum,
    frames: Vec<FrameTracker>,
}

impl PageTable {
    pub fn new() -> Self {
        let frame = frame_alloc().unwrap();
        PageTable {
            root_ppn: frame.ppn,
            frames: vec![frame],
        }
    }
}
```

- 数据结构
  - root_ppn：根节点的物理页号 `root_ppn`
    - 每个应用的地址空间都对应一个不同的多级页表，这也就意味这不同页表的起始地址（即页表根节点的地址）是不一样的。因此 `PageTable` 要保存它根节点的物理页号 `root_ppn` 作为页表唯一的区分标志。
  - 向量 `frames` ：
    - 以 `FrameTracker` 的形式保存了页表所有的节点（包括根节点）所在的物理页帧。
    - RAII：将这些 `FrameTracker` 的生命周期进一步绑定到 `PageTable` 下面。当 `PageTable` 生命周期结束后，向量 `frames` 里面的那些 `FrameTracker` 也会被回收，也就意味着存放多级页表节点的那些物理页帧被回收了。
- new方法
  - 新建一个 `PageTable` 的时候，它只需有一个根节点。为此我们需要分配一个物理页帧 `FrameTracker` 并挂在向量 `frames` 下，然后更新根节点的物理页号 `root_ppn` 。

#### 动态维护一个虚拟页号到页表项的映射

```rust
// os/src/mm/page_table.rs

impl PageTable {
    pub fn map(&mut self, vpn: VirtPageNum, ppn: PhysPageNum, flags: PTEFlags);
    pub fn unmap(&mut self, vpn: VirtPageNum);
}
```

- 当程序访问尚未映射的虚拟地址时（如第一次访问堆栈、mmap 区段、页面置换），页表必须 **动态插入或删除页表项（PTE）**。
- 操作系统需要动态维护一个虚拟页号到页表项的映射，支持插入/删除键值对，其方法签名如下：

```rust
// os/src/mm/page_table.rs

impl PageTable {
    pub fn map(&mut self, vpn: VirtPageNum, ppn: PhysPageNum, flags: PTEFlags);
    pub fn unmap(&mut self, vpn: VirtPageNum);
}
```

- 通过 `map` 方法来在多级页表中插入一个键值对，注意这里将物理页号 `ppn` 和页表项标志位 `flags` 作为不同的参数传入；
- 通过 `unmap` 方法来删除一个键值对，在调用时仅需给出作为索引的虚拟页号即可。



在上述操作的过程中，**内核需要能访问或修改多级页表节点的内容。**即在操作某个多级页表或管理物理页帧的时候，操作系统要能够读写与一个给定的物理页号对应的物理页帧上的数据

- 页表在访问前需要构造虚拟地址和物理地址的映射关系
  - 如果想要访问一个特定的物理地址 `pa` 所指向的内存上的数据，就需要 **构造** 对应的一个虚拟地址 `va` ，使得当前地址空间的页表存在映射 va→pa ，且页表项中的保护位允许这种访问方式。
- 恒等映射
  - 提前扩充多级页表维护的映射，让每个物理页帧的物理页号 `ppn` ，均存在一个对应的虚拟页号 `vpn` ，这需要建立一种映射关系。这里我们采用一种最简单的 **恒等映射** (Identical Mapping) ，即对于物理内存上的每个物理页帧，我们都在多级页表中用一个与其物理页号相等的虚拟页号来映射。

####  内核中访问物理页帧的方法

```Rust
// os/src/mm/address.rs

impl PhysPageNum {
    pub fn get_pte_array(&self) -> &'static mut [PageTableEntry] {
        let pa: PhysAddr = self.clone().into();
        unsafe {
            core::slice::from_raw_parts_mut(pa.0 as *mut PageTableEntry, 512)
        }
    }
    pub fn get_bytes_array(&self) -> &'static mut [u8] {
        let pa: PhysAddr = self.clone().into();
        unsafe {
            core::slice::from_raw_parts_mut(pa.0 as *mut u8, 4096)
        }
    }
    pub fn get_mut<T>(&self) -> &'static mut T {
        let pa: PhysAddr = self.clone().into();
        unsafe {
            (pa.0 as *mut T).as_mut().unwrap()
        }
    }
}
```

- 构造可变引用来直接访问一个物理页号 `PhysPageNum` 对应的物理页帧,不同的引用类型对应于物理页帧上的一种不同的内存布局
  - `get_pte_array` 返回的是一个**页表项定长数组**的可变引用，代表多级页表中的一个节点
  - `get_bytes_array` 返回的是一个**字节数组的可变引用**，可以以字节为粒度对物理页帧上的数据进行访问，前面进行数据清零就用到了这个方法
  - `get_mut` 是个泛型函数，可以获取一个恰好放在一个物理页帧开头的类型为 `T` 的数据的可变引用

- 实现

  - 先把物理页号转为物理地址 `PhysAddr` ，然后再转成 usize 形式的物理地址。接着，我们直接将它转为裸指针用来访问物理地址指向的物理内存。

  - 在分页机制开启前，这样做自然成立；而开启之后，虽然裸指针被视为一个虚拟地址，但是上面已经提到，基于恒等映射，虚拟地址会映射到一个相同的物理地址，因此在也是成立的。
  - 注意，我们在返回值类型上附加了静态生命周期泛型 `'static` ，这是为了绕过 Rust 编译器的借用检查，实质上可以将返回的类型也看成一个裸指针，因为它也只是标识数据存放的位置以及类型。但与裸指针不同的是，无需通过 `unsafe` 的解引用访问它指向的数据，而是可以像一个正常的可变引用一样直接访问。

> `from_raw_parts_mut` 
>
> - 是 Rust 中一个非常底层且强大的函数，可以在不经过 Rust 正常引用机制的情况下创建一个可变切片
>
> `static`
>
> - 使用这种写法的动机是：**在内核开发中，页帧一旦分配，生命周期就由内核自己管理，不会被自动释放**，所以允许 `'static`。

> **unsafe 真的就是“不安全”吗？**
>
> - 当我们在 Rust 中使用 unsafe 的时候，并不仅仅是为了绕过编译器检查，更是为了告知编译器和其他看到这段代码的程序员：“ **我保证这样做是安全的** ” 。尽管，严格的 Rust 编译器暂时还不能确信这一点。从规范 Rust 代码编写的角度，我们需要尽可能绕过 unsafe ，因为如果 Rust 编译器或者一些已有的接口就可以提供安全性，我们当然倾向于利用它们让我们实现的功能仍然是安全的，可以避免一些无谓的心智负担；反之，就只能使用 unsafe ，同时最好说明如何保证这项功能是安全的。
>
> - 第一个问题是 use-after-free 的问题，即是否存在 `get_*` 返回的引用存在期间被引用的物理页帧已被回收的情形；
>   - **手动完成对于get_*的资源的释放**
>     - 每一段 unsafe 代码，我们都需要认真考虑它会对其他无论是 unsafe 还是 safe 的代码造成的潜在影响。比如为了避免第一个问题，我们需要保证当完成物理页帧访问之后便立即回收掉 `get_*` 返回的引用，至少使它不能超出 `FrameTracker` 的生命周期；
>
> - 第二个问题则是注意到 `get_*` 返回的是可变引用，那么就需要考虑对物理页帧的访问读写冲突的问题。
>   - 考虑第二个问题，目前每个 `FrameTracker` 仅会出现一次（在它所属的进程中），因此它只会出现在一个上下文中，也就不会产生冲突。但是当内核态打开（允许）中断时，或内核支持在单进程中存在多个线程时，情况也许又会发生变化。
>
> 当编译器不能介入的时候，我们很难完美的解决这些问题。因此重新设计数据结构和接口，特别是考虑数据的所有权关系，将建模进行转换，使得 Rust 有能力检查我们的设计会是一种更明智的选择。这也可以说明为什么要尽量避免使用 unsafe 。事实上，我们**目前 `PhysPageNum::get_*` 接口并非一个好的设计，如果同学有兴趣可以试着对设计进行改良，让 Rust 编译器帮助我们解决上述与引用相关的问题。**

#### 建立和拆除虚实地址映射关系

`map` 和 `unmap` 方法

- 依赖于一个很重要的过程，在多级页表中找到一个虚拟地址对应的页表项。
- 找到之后，只要修改页表项的内容即可完成键值对的插入和删除。在寻找页表项的时候，可能出现页表的中间级节点还未被创建的情况，这个时候我们需要手动分配一个物理页帧来存放这个节点，并将这个节点接入到当前的多级页表的某级中。

```rust
// os/src/mm/address.rs

impl VirtPageNum {
    pub fn indexes(&self) -> [usize; 3] {
        let mut vpn = self.0;
        let mut idx = [0usize; 3];
        for i in (0..3).rev() {
            idx[i] = vpn & 511;
            vpn >>= 9;
        }
        idx
    }
}

// os/src/mm/page_table.rs

impl PageTable {
    fn find_pte_create(&mut self, vpn: VirtPageNum) -> Option<&mut PageTableEntry> {
        // 可以取出虚拟页号的三级页索引，并按照从高到低的顺序返回。可能是27也可能是53。
        let idxs = vpn.indexes();
        let mut ppn = self.root_ppn;
        let mut result: Option<&mut PageTableEntry> = None;
        for i in 0..3 {
            // 通过不同级的页索引9位在页表中找到页表项
            let pte = &mut ppn.get_pte_array()[idxs[i]];
            if i == 2 {
                result = Some(pte);
                break;
            }
            // 节点尚未创建新建一个节点
            if !pte.is_valid() {
                // 1.使用物理内存分配器分配一个物理页帧
                let frame = frame_alloc().unwrap();
                // 2.分配的物理页，通过物理页号和标志位更新页表项
                // 更新页表项的时候，不仅要更新物理页号，还要将标志位 V 置 1，不然硬件在查多级页表的时候，会认为这个页表项不合法，从而触发 Page Fault 而不能向下走。
                *pte = PageTableEntry::new(frame.ppn, PTEFlags::V);
                // 放入frames中方便后面自动回收
                self.frames.push(frame);
            }
            ppn = pte.ppn();
        }
        result
    }
    // 和find_pte_create类似，但不会创建页表项，找不到就返回None
    // 它不会尝试对页表本身进行修改，但是注意它返回的参数类型是页表项的可变引用，也即它允许我们修改页表项。
    fn find_pte(&self, vpn: VirtPageNum) -> Option<&mut PageTableEntry> {
        let idxs = vpn.indexes();
        let mut ppn = self.root_ppn;
        let mut result: Option<&mut PageTableEntry> = None;
        for i in 0..3 {
            let pte = &mut ppn.get_pte_array()[idxs[i]];
            if i == 2 {
                result = Some(pte);
                break;
            }
            if !pte.is_valid() {
                return None;
            }
            ppn = pte.ppn();
        }
        result
    }
}
```

于是， `map/unmap` 就非常容易实现了：

```rsut
// os/src/mm/page_table.rs

impl PageTable {
    pub fn map(&mut self, vpn: VirtPageNum, ppn: PhysPageNum, flags: PTEFlags) {
        let pte = self.find_pte_create(vpn).unwrap();
        assert!(!pte.is_valid(), "vpn {:?} is mapped before mapping", vpn);
        // 修改页表项的ppn
        *pte = PageTableEntry::new(ppn, flags | PTEFlags::V);
    }
    pub fn unmap(&mut self, vpn: VirtPageNum) {
        let pte = self.find_pte(vpn).unwrap();
        assert!(pte.is_valid(), "vpn {:?} is invalid before unmapping", vpn);
        *pte = PageTableEntry::empty();
    }
}
```

`PageTable` 提供一种类似 MMU 操作的手动查页表的方法：

```Rust
// os/src/mm/page_table.rs

impl PageTable {
    /// Temporarily used to get arguments from user space.
    pub fn from_token(satp: usize) -> Self {
        Self {
            root_ppn: PhysPageNum::from(satp & ((1usize << 44) - 1)),
            frames: Vec::new(),
        }
    }
    pub fn translate(&self, vpn: VirtPageNum) -> Option<PageTableEntry> {
        self.find_pte(vpn)
            .map(|pte| {pte.clone()})
    }
}
```

- from_token
  - 它仅有一个从传入的 `satp` token 中得到的多级页表根节点的物理页号，它的 `frames` 字段为空，也即不实际控制任何资源；
- 第 11 行的 `translate` 调用 `find_pte` 来实现，如果能够找到页表项，那么它会将页表项拷贝一份并返回，否则就返回一个 `None` 。

## 内核与应用的地址空间

操作系统通过对不同页表的管理，来完成对不同应用和操作系统自身所在的虚拟内存，以及虚拟内存与物理内存映射关系的全面管理。这种管理是建立在 **地址空间** 的抽象上，用来表明正在运行的应用或内核自身所在执行环境中的可访问的内存空间。

### 实现地址空间抽象

#### 逻辑段：一段连续地址的虚拟内存

以逻辑段 `MapArea` 为单位描述一段连续地址的虚拟内存,地址区间中的一段实际可用（即 MMU 通过查多级页表可以正确完成地址转换）的地址连续的虚拟地址区间，该区间内包含的所有虚拟页面都以一种相同的方式映射到物理页帧，具有可读/可写/可执行等属性。

```rust
// os/src/mm/memory_set.rs

pub struct MapArea {
    vpn_range: VPNRange,
    data_frames: BTreeMap<VirtPageNum, FrameTracker>,
    map_type: MapType,
    map_perm: MapPermission,
}
```

- VPNRange：虚拟页号的连续区间，表示该逻辑段在地址区间中的位置和长度
  - 是一个迭代器，可以使用 Rust 的语法糖 for-loop 进行迭代。

- MapType：描述该逻辑段内的所有虚拟页面映射到物理页帧的同一种方式，它是一个枚举类型，在内核当前的实现中支持两种方式：

  - `Identical`：恒等映射方式
  - `Framed`：对于每个虚拟页面都有一个新分配的物理页帧与之对应，虚地址与物理地址的映射关系是相对随机的。
    - 恒等映射方式主要是用在启用多级页表之后，内核仍能够在虚存地址空间中访问一个特定的物理地址指向的物理内存。

- data_frames：MapType::Framed时保存了该逻辑段内的每个虚拟页面和它被映射到的物理页帧 `FrameTracker` 的一个键值对容器 `BTreeMap`

  - 存放实际内存数据而不是作为多级页表中的中间节点

- MapPermission：控制该逻辑段的访问方式，它是页表项标志位 `PTEFlags` 的一个子集，仅保留 U/R/W/X 四个标志位，因为其他的标志位仅与硬件的地址转换机制细节相关，这样的设计能避免引入错误的标志位。

  - ```rust
    // os/src/mm/memory_set.rs
    
    bitflags! {
        pub struct MapPermission: u8 {
            const R = 1 << 1;
            const W = 1 << 2;
            const X = 1 << 3;
            const U = 1 << 4;
        }
    }
    ```

#### 地址空间：一系列有关联的逻辑段

**地址空间** 是一系列有关联的不一定连续的逻辑段，这种关联一般是指**这些逻辑段**组成的虚拟内存空间与一个运行的**程序绑定**(目前把一个运行的程序称为任务，后续会称为进程），即这个运行的程序对代码和数据的直接访问范围限制在它关联的虚拟地址空间之内。

```Rust
// os/src/mm/memory_set.rs

pub struct MemorySet {
    page_table: PageTable,
    areas: Vec<MapArea>,
}
```

- 组成：

  -  `PageTable` ：下挂着所有多级页表的节点所在的物理页帧

  -  `MapArea` ：下则挂着对应逻辑段中的数据所在的物理页帧，这两部分合在一起构成了一个地址空间所需的所有物理页帧。

- RAII :这同样是一种 RAII 风格，当一个地址空间 `MemorySet` 生命周期结束后，这些物理页帧都会被回收。

地址空间`MemorySet` 的方法如下：

```rust
// os/src/mm/memory_set.rs

impl MemorySet {
    // 新建一个空的地址空间
    pub fn new_bare() -> Self {
        Self {
            page_table: PageTable::new(),
            areas: Vec::new(),
        }
    }
    fn push(&mut self, mut map_area: MapArea, data: Option<&[u8]>) {
        map_area.map(&mut self.page_table);
        if let Some(data) = data {
            map_area.copy_data(&self.page_table, data);
        }
        self.areas.push(map_area);
    }
    /// Assume that no conflicts.
    pub fn insert_framed_area(
        &mut self,
        start_va: VirtAddr, end_va: VirtAddr, permission: MapPermission
    ) {
        self.push(MapArea::new(
            start_va,
            end_va,
            MapType::Framed,
            permission,
        ), None);
    }
    pub fn new_kernel() -> Self;
    /// Include sections in elf and trampoline and TrapContext and user stack,
    /// also returns user_sp and entry point.
    pub fn from_elf(elf_data: &[u8]) -> (Self, usize, usize);
}
```

- 第 4 行， `new_bare` 方法可以新建一个空的地址空间；
- 第 10 行， `push` 方法可以在当前地址空间插入一个新的逻辑段 `map_area` ，如果它是以 `Framed` 方式映射到物理内存，还可以可选地在那些被映射到的物理页帧上写入一些初始化数据 `data` ；
- 第 18 行， `insert_framed_area` 方法调用 `push` ，可以在当前地址空间插入一个 `Framed` 方式映射到物理内存的逻辑段。注意该方法的调用者要保证同一地址空间内的任意两个逻辑段不能存在交集，从后面即将分别介绍的内核和应用的地址空间布局可以看出这一要求得到了保证；
- 第 29 行， `new_kernel` 可以生成内核的地址空间；具体实现将在后面讨论；
- 第 32 行， `from_elf` 分析应用的 ELF 文件格式的内容，解析出各数据段并生成对应的地址空间；具体实现将在后面讨论。

##### 插入逻辑段也需要维护页表中的映射关系

在地址空间中插入一个逻辑段 `MapArea` 的时候，需要同时维护地址空间的多级页表 `page_table` 记录的虚拟页号到页表项的映射关系，也需要用到这个映射关系来找到向哪些物理页帧上拷贝初始数据。

```Rust
// os/src/mm/memory_set.rs

impl MapArea {
    pub fn new(
        start_va: VirtAddr,
        end_va: VirtAddr,
        map_type: MapType,
        map_perm: MapPermission
    ) -> Self {
        let start_vpn: VirtPageNum = start_va.floor();
        let end_vpn: VirtPageNum = end_va.ceil();
        Self {
            vpn_range: VPNRange::new(start_vpn, end_vpn),
            data_frames: BTreeMap::new(),
            map_type,
            map_perm,
        }
    }
    pub fn map(&mut self, page_table: &mut PageTable) {
        for vpn in self.vpn_range {
            self.map_one(page_table, vpn);
        }
    }
    pub fn unmap(&mut self, page_table: &mut PageTable) {
        for vpn in self.vpn_range {
            self.unmap_one(page_table, vpn);
        }
    }
    /// data: start-aligned but maybe with shorter length
    /// assume that all frames were cleared before
    pub fn copy_data(&mut self, page_table: &PageTable, data: &[u8]) {
        assert_eq!(self.map_type, MapType::Framed);
        let mut start: usize = 0;
        let mut current_vpn = self.vpn_range.get_start();
        let len = data.len();
        loop {
            let src = &data[start..len.min(start + PAGE_SIZE)];
            let dst = &mut page_table
                .translate(current_vpn)
                .unwrap()
                .ppn()
                .get_bytes_array()[..src.len()];
            dst.copy_from_slice(src);
            start += PAGE_SIZE;
            if start >= len {
                break;
            }
            current_vpn.step();
        }
    }
}
```

- 第 4 行的 `new` 方法可以新建一个逻辑段结构体，注意传入的起始/终止虚拟地址会分别被下取整/上取整为虚拟页号并传入迭代器 `vpn_range` 中；

- 第 19 行的 `map` 和第 24 行的 `unmap` 可以将当前逻辑段到物理内存的映射从传入的该逻辑段所属的地址空间的多级页表中加入或删除。可以看到它们的实现是遍历逻辑段中的所有虚拟页面，并以每个虚拟页面为单位依次在多级页表中进行键值对的插入或删除，分别对应 `MapArea` 的 `map_one` 和 `unmap_one` 方法，我们后面将介绍它们的实现；

- 第 31 行的 `copy_data` 方法将切片 `data` 中的数据拷贝到当前逻辑段实际被内核放置在的各物理页帧上，从而在地址空间中通过该逻辑段就能访问这些数据。调用它的时候需要满足：切片 `data` 中的数据大小不超过当前逻辑段的总大小，且切片中的数据会被对齐到逻辑段的开头，然后逐页拷贝到实际的物理页帧。

  从第 36 行开始的循环会遍历每一个需要拷贝数据的虚拟页面，在数据拷贝完成后会在第 48 行通过调用 `step` 方法，该方法来自于 `os/src/mm/address.rs` 中为 `VirtPageNum` 实现的 `StepOne` Trait，感兴趣的同学可以阅读代码确认其实现。

  每个页面的数据拷贝需要确定源 `src` 和目标 `dst` 两个切片并直接使用 `copy_from_slice` 完成复制。当确定目标切片 `dst` 的时候，第 39 行从传入的当前逻辑段所属的地址空间的多级页表中，手动查找迭代到的虚拟页号被映射到的物理页帧，并通过 `get_bytes_array` 方法获取该物理页帧的字节数组型可变引用，最后再获取它的切片用于数据拷贝。

接下来介绍对逻辑段中的单个虚拟页面进行映射/解映射的方法 `map_one` 和 `unmap_one` 。显然它们的实现取决于当前逻辑段被映射到物理内存的方式：

```rust
// os/src/mm/memory_set.rs

impl MapArea {
    pub fn map_one(&mut self, page_table: &mut PageTable, vpn: VirtPageNum) {
        let ppn: PhysPageNum;
        match self.map_type {
            MapType::Identical => {
                ppn = PhysPageNum(vpn.0);
            }
            MapType::Framed => {
                let frame = frame_alloc().unwrap();
                ppn = frame.ppn;
                self.data_frames.insert(vpn, frame);
            }
        }
        let pte_flags = PTEFlags::from_bits(self.map_perm.bits).unwrap();
        page_table.map(vpn, ppn, pte_flags);
    }
    pub fn unmap_one(&mut self, page_table: &mut PageTable, vpn: VirtPageNum) {
        match self.map_type {
            MapType::Framed => {
                self.data_frames.remove(&vpn);
            }
            _ => {}
        }
        page_table.unmap(vpn);
    }
}
```

- 对于第 4 行的 `map_one` 来说，在虚拟页号 `vpn` 已经确定的情况下，它需要知道要将一个怎么样的页表项插入多级页表。页表项的标志位来源于当前逻辑段的类型为 `MapPermission` 的统一配置，只需将其转换为 `PTEFlags` ；而页表项的物理页号则取决于当前逻辑段映射到物理内存的方式：

  - 当以恒等映射 `Identical` 方式映射的时候，物理页号就等于虚拟页号；
  - 当以 `Framed` 方式映射时，需要分配一个物理页帧让当前的虚拟页面可以映射过去，此时页表项中的物理页号自然就是 这个被分配的物理页帧的物理页号。此时还需要将这个物理页帧挂在逻辑段的 `data_frames` 字段下。

  当确定了页表项的标志位和物理页号之后，即可调用多级页表 `PageTable` 的 `map` 接口来插入键值对。

- 对于第 19 行的 `unmap_one` 来说，基本上就是调用 `PageTable` 的 `unmap` 接口删除以传入的虚拟页号为键的键值对即可。然而，当以 `Framed` 映射的时候，不要忘记同时将虚拟页面被映射到的物理页帧 `FrameTracker` 从 `data_frames` 中移除，这样这个物理页帧才能立即被回收以备后续分配。

### 内核地址空间

- 隔离：
  - 地址空间抽象的重要意义在于 **隔离** (Isolation) ，当内核让应用执行前，内核需要控制 MMU 使用这个应用的多级页表进行地址转换。
  - 由于每个应用地址空间在创建的时候也顺带设置好了多级页表，使得只有那些存放了它的代码和数据的物理页帧能够通过该多级页表被映射到，这样它就只能访问自己的代码和数据而无法触及其他应用或内核的内容。

- 启用分页模式下，内核代码的访存地址也会被视为一个虚拟地址并需要经过 MMU 的地址转换，因此我们也需要为内核对应构造一个地址空间，它除了仍然需要允许内核的各数据段能够被正常访问之后，还需要包含所有应用的内核栈以及一个 **跳板** (Trampoline) 。我们会在本章的后续部分再深入介绍 [跳板的实现](https://rcore-os.cn/rCore-Tutorial-Book-v3/chapter4/6multitasking-based-on-as.html#term-trampoline) 。
