docker attach 用于连接到一个正在运行的容器，主要作用是 访问容器的主进程（PID=1）的标准输入输出流
``` shell

docker attach <container_id>

```
由于 attach 是接管了 PID=1 的进程，因此如果这个进程是守护进程，

那么 attach 退出后，容器也会退出。所以一般不推荐使用 attach 命令。

而是使用 `docker exec` 命令来连接容器。