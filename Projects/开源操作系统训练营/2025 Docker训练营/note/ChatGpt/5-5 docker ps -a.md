```
➜  /workspace git:(main) docker ps -a
CONTAINER ID   IMAGE     COMMAND   CREATED         STATUS                     PORTS     NAMES
7ed4e65f1816   alpine    "ls -a"   5 minutes ago   Exited (0) 5 minutes ago             romantic_darwin
```
- 其中 `-a` 是 `--all` 的缩写，表示显示 **所有容器**（而不仅仅是正在运行的容器）。

| 字段名              | 示例值                        | 含义说明                                              |
| ---------------- | -------------------------- | ------------------------------------------------- |
| **CONTAINER ID** | `7ed4e65f1816`             | 容器的唯一 ID（前缀）                                      |
| **IMAGE**        | `alpine`                   | 创建容器所用的镜像名                                        |
| **COMMAND**      | `"ls -a"`                  | 容器启动时执行的命令（带引号）                                   |
| **CREATED**      | `5 minutes ago`            | 容器的创建时间                                           |
| **STATUS**       | `Exited (0) 5 minutes ago` | 容器的当前状态，此处表示“5 分钟前正常退出（exit code 0）”              |
| **PORTS**        | 空                          | 此容器没有暴露任何端口                                       |
| **NAMES**        | `romantic_darwin`          | Docker 自动分配的容器名称，如果未手动指定，Docker 会用形容词+科学家姓氏组合随机生成 |
