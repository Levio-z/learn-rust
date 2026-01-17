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
### `libc::mmap` 解释

这一行代码调用的是 **Linux/Unix 系统调用 `mmap`**，它在 Rust 中通过 `libc` 库封装成了一个函数。作用是 **在进程地址空间映射一段内存**，可以是文件内容或匿名内存。

---

#### 函数原型（C 语言）

```c
void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset);
```

#### Rust 对应写法

```rust
let ptr = libc::mmap(
    ptr::null_mut(),              // addr: 由内核选择起始地址
    aligned_len as libc::size_t,  // length: 映射长度
    prot,                         // prot: 访问权限，例 PROT_READ | PROT_WRITE
    flags,                        // flags: 映射类型，例 MAP_SHARED / MAP_PRIVATE
    file,                         // fd: 文件描述符
    aligned_offset as libc::off_t // offset: 文件偏移
);
```

---

#### 参数解析

| 参数       | Rust 传入值          | 说明                                                                                                                  |
| -------- | ----------------- | ------------------------------------------------------------------------------------------------------------------- |
| `addr`   | `ptr::null_mut()` | 内核选择映射的起始地址（如果想指定地址可以传指针，但通常为 null）                                                                                 |
| `length` | `aligned_len`     | 映射的字节长度，通常要按页对齐                                                                                                     |
| `prot`   | `prot`            | 内存访问权限，常见值：`PROT_READ`, `PROT_WRITE`, `PROT_EXEC`                                                                   |
| `flags`  | `flags`           | 映射类型，常见值：`MAP_SHARED`（写入文件）、`MAP_PRIVATE`（写入内存不影响文件）、`MAP_ANONYMOUS`（匿名映射，不关联文件）[libc-mmap-选项分析](libc-mmap-选项分析.md) |
| `fd`     | `file`            | 文件描述符，若匿名映射传 `-1`                                                                                                   |
| `offset` | `aligned_offset`  | 文件映射偏移量（必须是页对齐）                                                                                                     |
|          |                   |                                                                                                                     |

---

#### 返回值

- 成功：返回映射的起始地址 `*mut c_void`（裸指针）。
    
- 失败：返回 `MAP_FAILED`（通常是 `-1`），可以通过 `errno` 获取错误信息。
    

Rust 中的处理：

```rust
if ptr == libc::MAP_FAILED {
    Err(io::Error::last_os_error())
} else {
    Ok(MmapInner { ptr: ptr.offset(alignment as isize), len })
}
```

- `ptr.offset(alignment)`：调整偏移，去掉页对齐引入的额外空间。
    
- `len`：保持用户请求的长度。
    

---

#### 原理

`mmap` 本质上是 **虚拟内存映射**：

1. 把文件或匿名内存映射到进程虚拟地址空间。
    
2. CPU 访问映射内存时会触发页表映射，像访问普通数组一样读写。
    
3. 可以用 `MAP_SHARED` 让多个进程共享同一块内存，或者 `MAP_PRIVATE` 创建副本。
    






## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
