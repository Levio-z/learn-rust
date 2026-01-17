---
tags:
  - note
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
### `MmapInner::new` 解释

#### 定义与作用

这个函数是 Rust 中对 **`mmap` 系统调用** 的封装，用于在用户空间映射文件或匿名内存到进程的地址空间。`MmapInner` 封装了映射后的指针和长度，提供安全访问接口。

- **函数签名**：
    

```rust
fn new(len: usize, prot: libc::c_int, flags: libc::c_int, file: RawFd, offset: u64) -> io::Result<MmapInner>
```

- **参数**：
    
    - `len`: 映射的字节长度。
        
    - `prot`: 映射的内存保护属性，例如 `PROT_READ | PROT_WRITE`。
        
    - `flags`: 映射类型，例如 `MAP_SHARED` 或 `MAP_PRIVATE`。
        
    - `file`: 文件描述符，如果是匿名映射可以传 `-1`。
        
    - `offset`: 文件映射的起始偏移量（单位字节）。
        
- **返回值**：
    
    - `Ok(MmapInner)`：成功映射。
        
    - `Err(io::Error)`：映射失败。
        

---

#### 核心逻辑

1. **对齐处理**：
    

```rust
let alignment = offset % page_size() as u64;
let aligned_offset = offset - alignment;
let aligned_len = len + alignment as usize;
```

- `mmap` 的偏移量必须是 **页大小** 的整数倍（通常 4KB）。
    
- 计算：
    
    - `alignment`：偏移量相对于页大小的余数。
    - `aligned_offset`：对齐后的偏移量。
    - `aligned_len`：映射长度增加偏移对齐的长度，保证整个请求都被覆盖。
    
例如：

- `page_size() = 4096`
- `offset = 5000`
- `alignment = 5000 % 4096 = 904`
- `aligned_offset = 5000 - 904 = 4096`    
- len + alignment as usize长度就是把前面的映射给加上
	- 实际用户想访问的范围是 `[5000, 6000)`。**如果不加 `alignment`，最后 104 个字节（6000 - 5096）就会访问越界**。
```
文件偏移:   4096           5000             6000
            |--------------|----------------|
mmap映射:   [4096 ------------------------ 6000]
返回 ptr:           ↑
                 用户访问起点 (5000)
长度:                 1000
```
2. **长度检查**：
    

```rust
if aligned_len == 0 {
    return Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        "memory map must have a non-zero length",
    ));
}
```

- 防止零长度映射，某些环境（如 QEMU）可能会导致段错误（segfault）。


3. **调用 `libc::mmap`**：
    

```rust
let ptr = libc::mmap(
    ptr::null_mut(),
    aligned_len as libc::size_t,
    prot,
    flags,
    file,
    aligned_offset as libc::off_t,
);
```

- `ptr::null_mut()`：由内核选择映射地址。
    
- `aligned_len`：映射长度。
    
- `prot`、`flags`：控制映射权限与类型。
    
- `file`、`aligned_offset`：指定映射文件及起始偏移。
    

4. **检查映射结果**：
    

```rust
if ptr == libc::MAP_FAILED {
    Err(io::Error::last_os_error())
} else {
    Ok(MmapInner {
        ptr: ptr.offset(alignment as isize),
        len: len,
    })
}
```

- 如果返回 `MAP_FAILED`，则映射失败，用 `io::Error::last_os_error()` 获取 errno。
    
- 返回 `MmapInner`：
    
    - `ptr`：调整后的指针，去掉偏移对齐的多余部分。
        
    - `len`：原始长度。
        

---

#### 原理解析

1. **内存对齐**：`mmap` 必须按页对齐，函数内部做了偏移补偿。
    
2. **安全封装**：直接操作裸指针，但封装在 Rust 类型中，提供了长度信息，方便后续安全访问。
    
3. **跨平台考虑**：对零长度映射、QEMU 等特殊情况进行了保护。
    

---

#### 使用场景

- 映射大文件到内存（如数据库、视频处理）。
    
- 共享内存实现进程间通信（`MAP_SHARED`）。
    
- 分配匿名内存而不经过堆（`MAP_ANONYMOUS`）。
    

---

#### 总结与方法论

1. **总结**：
    
    - `MmapInner::new` 是 mmap 的安全封装。
        
    - 内部处理了页对齐、长度检查、错误处理。
        
    - 返回的指针已经调整到用户请求的起始位置。
        
2. **学习方法**：
    
    - 熟悉 `mmap` 系统调用文档。
        
    - 理解页对齐和偏移量调整的原理。
        
    - 在 Rust 中练习裸指针操作与安全封装。
        
3. **重点**：
    
    - 页对齐逻辑。
        
    - 指针偏移调整。
        
    - 错误处理与跨平台兼容性。
        
4. **练习**：
    
    - 用 `MmapInner::new` 映射一个文件并读写数据。
        
    - 尝试映射带有不同偏移量的文件片段，观察指针对齐。
        
    - 实现一个安全 wrapper，允许切片访问映射的内存。
        

---

如果你需要，我可以画一张 **偏移与对齐示意图**，直观显示 `aligned_offset`、`alignment` 与返回 `ptr` 的关系。你希望我画吗？


## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
