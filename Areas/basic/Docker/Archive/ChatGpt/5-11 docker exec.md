``` shell

docker exec -it <container_id> /bin/sh

```
此时进入到容器中使用 `ps -a` 命令可以看到容器中存在两个进程，其中 PID=1 的进程为 /bin/sh，

而另一个 /bin/sh 进程则是我们通过 exec 命令启动的，这个进程退出不会影响 PID=1 的进程，也就不会导致容器的退出。