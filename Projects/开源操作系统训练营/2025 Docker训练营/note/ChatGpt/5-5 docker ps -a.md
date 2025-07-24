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
### 命令格式
```
docker ps [OPTIONS]
```
- 不加 `-a`，**只列出正在运行的容器**。
- 加 `-a`，**列出所有容器**。
### 常用选项

| 选项            | 说明                        |
| ------------- | ------------------------- |
| `-a`, `--all` | 显示所有容器，包括停止状态的            |
| `-q`          | 只输出容器 ID                  |
| `--filter`    | 按条件过滤容器，如 `status=exited` |
| `--format`    | 自定义输出格式，支持 Go 模板          |
| `--no-trunc`  | 不截断输出，显示完整的容器 ID 和命令      |
| `-n=?`        | 仅显示最近创建的 n 个容器            |
| `-l`          | 显示最新创建的一个容器               |

### 示例
#### --filter
过滤退出状态的容器
```
➜  /workspace git:(main) docker ps --filter=status=exited
CONTAINER ID   IMAGE     COMMAND       CREATED         STATUS                     PORTS     NAMES
6346ec7a911a   ubuntu    "/bin/bash"   3 minutes ago   Exited (0) 3 minutes ago             youthful_booth
```