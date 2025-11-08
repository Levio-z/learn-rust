```c
#include "apue.h"
#include <errno.h>
#include <fcntl.h>

char buf[500000];

/* 设置文件描述符标志位（如 O_NONBLOCK） */
void set_fl(int fd, int flags)
{
    int val;

    if ((val = fcntl(fd, F_GETFL, 0)) < 0)
        err_sys("fcntl F_GETFL error");

    val |= flags;  /* 设置标志位 */
    if (fcntl(fd, F_SETFL, val) < 0)
        err_sys("fcntl F_SETFL error");
}

/* 清除文件描述符标志位 */
void clr_fl(int fd, int flags)
{
    int val;

    if ((val = fcntl(fd, F_GETFL, 0)) < 0)
        err_sys("fcntl F_GETFL error");

    val &= ~flags; /* 清除指定标志位 */
    if (fcntl(fd, F_SETFL, val) < 0)
        err_sys("fcntl F_SETFL error");
}

int main(void)
{
    int ntowrite, nwrite;
    char *ptr;

    /* 从标准输入读入数据 */
    ntowrite = read(STDIN_FILENO, buf, sizeof(buf));
    fprintf(stderr, "read %d bytes\n", ntowrite);

    /* 将标准输出设为非阻塞模式 */
    set_fl(STDOUT_FILENO, O_NONBLOCK);

    ptr = buf;
    while (ntowrite > 0) {
        errno = 0;
        nwrite = write(STDOUT_FILENO, ptr, ntowrite);
        fprintf(stderr, "nwrite = %d, errno = %d\n", nwrite, errno);

        if (nwrite > 0) {
            ptr += nwrite;
            ntowrite -= nwrite;
        } else if (nwrite < 0 && errno != EAGAIN) {
            err_sys("write error");
        }
        /* 若 errno == EAGAIN 表示写缓冲区满，重试即可 */
    }

    /* 恢复标准输出为阻塞模式 */
    clr_fl(STDOUT_FILENO, O_NONBLOCK);

    exit(0);
}


```

#### 1. 功能定义

该程序用于**演示非阻塞 I/O（Nonblocking I/O）**的行为：

- 从标准输入读取一大块数据（500000 字节），
    
- 然后以非阻塞模式不断向标准输出写出。
    

#### 2. 核心机制

- **`fcntl`** 系统调用用于修改文件描述符的状态标志。
    
    - `F_GETFL`: 获取当前状态标志。
        
    - `F_SETFL`: 设置新状态标志。
        
- **`O_NONBLOCK`** 使 `write()` 在写缓冲区满时立即返回 `-1` 并设置 `errno = EAGAIN`，而不是阻塞等待。

### 测试结果


终端驱动程序一次能接受的数据量随系统而变：这解释了为什么你的非阻塞 write 示例程序在不同机器上运行时，返回  的时间点和频率会不一样。
• write 成功写入的字节数（即缓冲区容量）取决于底层设备驱动程序（例如终端驱动程序）的内部缓冲区大小。

在该系统上，errno 值35 对应的是EAGAIN。终端驱动程序一次能接受的数据量随系统而变。具体结果还会因登录系统时所使用的方式的不同而不同：在系统控制台上登录、在硬接线的终端上登录或用伪终端在网络连接上登录。如果你在终端上运行一个窗口系统，那么也是经由伪终端设备与系统交互。

在此实例中，程序发出了9 000多个write调用，但是只有500个真正输出了数据，其余的都只返回了错误。这种形式的循环称为轮询，在多用户系统上用它会浪费CPU时间。14.4节将介绍非阻塞描述符的I/O多路转接，这是进行这种操作的一种比较有效的方法。