
`<cstdio>` 是 C++ 中的一个标准头文件，用于引入 C 语言风格的输入/输出（I/O）函数，即所谓的 **C-style I/O**。这个头文件是从 C 的 `<stdio.h>` 继承而来，属于 C++ 标准库的 C 兼容部分。

---

### ✅ 一、`<cstdio>` 的定义与作用

#### 1\. 定义

```cpp
#include <cstdio>
```

这是 C++ 标准库提供的对 `<stdio.h>` 的 C++ 封装，实质上是：

```cpp
extern "C" {
    #include <stdio.h>
}
```

#### 2\. 作用

提供 **C 标准输入/输出函数**，包括但不限于：

-   标准输入输出
    
    -   `printf`, `scanf`
        
-   文件操作
    
    -   `fopen`, `fclose`, `fread`, `fwrite`
        
-   格式化 I/O
    
    -   `sprintf`, `snprintf`
        
-   字符操作
    
    -   `fgetc`, `fputc`, `getc`, `putc`
        
-   缓冲区刷新
    
    -   `fflush`
        

---

### ✅ 二、核心函数/宏总览

| 类型 | 关键函数/宏 | 说明 |
| --- | --- | --- |
| 标准流 | `stdin`, `stdout`, `stderr` | 分别表示标准输入、标准输出、标准错误输出 |
| 基本 I/O | `printf`, `scanf` | 格式化输出与输入 |
| 字符 I/O | `fgetc`, `fputc`, `getchar`, `putchar` | 单字符操作 |
| 字符串 I/O | `fgets`, `fputs` | 读取/写入字符串 |
| 文件操作 | `fopen`, `fclose`, `fseek`, `ftell` ,`fprintf`| 文件读写与定位 |
| 缓冲控制 | `fflush`, `setvbuf` | 刷新或设置缓冲区 |
| 错误处理 | `perror`, `ferror`, `clearerr` | 错误检测与处理 |

---

### ✅ 三、使用场景与举例

#### 📌 示例 1：标准输出

```cpp
#include <cstdio>

int main() {
    printf("Hello, C-style I/O!\n");
    return 0;
}
```

#### 📌 示例 2：文件读写

```cpp
#include <cstdio>

int main() {
    FILE* file = fopen("example.txt", "w");
    if (file) {
        fprintf(file, "Writing to file using <cstdio>\n");
        fclose(file);
    }
    return 0;
}
```
- `fopen`：打开或创建一个文件，并返回一个指向该文件的 `FILE*` 类型指针。
- 参数说明：
    - `"example.txt"`：文件名。若不存在则会被创建。
    - `"w"`：写入模式（write mode）：
        - 若文件存在，则清空原有内容。
        - 若文件不存在，则创建新文件。
- 底层：
	- 申请文件描述符为文件创建缓冲区
- 返回值：
    - 成功 → 有效的 `FILE*`
    - 失败 → `NULL`（例如：没有权限、路径错误等）
- `fprintf()`：向指定文件流写入格式化字符串（类似 `printf()`）。
	- 底层：写入缓存区（内存），不立刻写磁盘
-  关闭打开的文件，释放资源。
	- 非常重要，否则缓冲区内容**可能不会被完全写入**磁盘（因 `fprintf` 是缓冲写入）。
	- 自动触发 `fflush(file)` → 刷新缓冲区。
	- 底层
		- 调用flush写入磁盘
		- 关闭文件描述符
#### 📌 示例 3：标准流的使用

```cpp
#include <cstdio>

int main() {
    fprintf(stdout, "This goes to stdout\n");
    fprintf(stderr, "This goes to stderr\n");
    return 0;
}
```

---

### ✅ 四、与 C++ 风格的区别（对比）

| 特性 | `<cstdio>`（C 风格） | `<iostream>`（C++ 风格） |
| --- | --- | --- |
| 输入 | `scanf()` | `std::cin` |
| 输出 | `printf()` | `std::cout`, `std::cerr` |
| 类型安全 | ❌ 否，需要格式控制 | ✅ 是，重载运算符 |
| 可扩展性 | 差 | 强，可重载 `<<` |
| 性能 | 更接近系统层 | 稍慢（但可优化） |
| 风格 | 过程式 | 面向对象 |

---

### ✅ 五、原理与底层机制简析

-   所有 C-style I/O 操作底层都依赖于 **缓冲区 I/O**（Buffered I/O）。
    
-   `FILE*` 是对文件流的抽象，`fopen` 打开文件时，会在内部初始化缓冲区。
    
-   `printf` 等格式化函数使用可变参数列表（`va_list`）和格式字符串进行拼接。
    
-   标准流（`stdin`, `stdout`, `stderr`）在程序启动时由运行时自动初始化。
    

---

### ✅ 六、延伸知识

-   `fflush(stdout)` 常用于强制立即输出缓存内容（特别是在调试时）。
    
-   `setvbuf()` 可用于设置输出缓冲模式（全缓冲、行缓冲、无缓冲）。
    
-   C++ 中不建议混用 `<cstdio>` 和 `<iostream>`，否则会有缓冲顺序问题，可用：
    
    ```cpp
    std::ios::sync_with_stdio(false); // 关闭同步以提升性能，但需谨慎
    ```
    

---

### ✅ 七、总结（速记）

| 你需要 | 使用 |
| --- | --- |
| 兼容旧 C 代码 | `<cstdio>` |
| 写日志、调试输出 | `printf`, `fprintf(stderr, …)` |
| 文件读写（简单场景） | `fopen` / `fread` 等 |
| 高性能输出 | `printf`（配合 `fflush`） |
| 推荐写现代 C++ | `<iostream>` 更安全更可维护 |

---