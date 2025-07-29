
- 这是一个非常实用的 Docker 命令，用于查看镜像的默认启动命令（即 `CMD` 指令的值）

|部分|说明|
|---|---|
|`docker inspect alpine`|查看镜像 `alpine` 的元数据（通常是 JSON 格式）|
|`--format='{{.Config.Cmd}}'`|使用 Go 模板语法，格式化输出其中 `.Config.Cmd` 字段（即镜像中定义的默认命令）|
### 🧩 示例输出与解释：

#### 示例输出（可能是）：

```bash
[/bin/sh]
```

这表示 **该镜像默认执行 `/bin/sh` 命令**。也就是说：

* 如果你运行：
    
    ```bash
    docker run -it alpine
    ```
    
    没指定命令时，它等价于：
    
    ```bash
    docker run -it alpine /bin/sh
    ```
    

* * *

### 🔍 什么是 `.Config.Cmd`？

* `.Config.Cmd` 是镜像 Dockerfile 中的 `CMD` 指令所定义的默认命令参数；
    
* 它只有在运行容器时 **未指定额外命令** 时才会被调用；
    
* `CMD` 不等同于 `ENTRYPOINT`，但两者组合决定容器的启动行为。

### 🧪 举个具体例子：

比如你写了一个 Dockerfile：

```dockerfile
FROM alpine
CMD ["echo", "hello world"]
```

构建镜像：

```bash
docker build -t myhello .
```

检查：

```bash
docker inspect myhello --format='{{.Config.Cmd}}'
```

输出：

```bash
[echo hello world]
```

这说明：

```bash
docker run myhello
```

相当于运行：

```bash
docker run myhello echo hello world
```

* * *

### 🧠 扩展知识：还有哪些 `.Config.*` 可查看？

你可以通过如下命令，探索更多字段：

```bash
docker inspect alpine
```

或用格式化方式查看：

```bash
docker inspect alpine --format='{{json .Config}}'
```

其中可能包括：

| 字段名 | 含义 |
| --- | --- |
| `Cmd` | 默认命令（如 `/bin/sh`） |
| `Entrypoint` | 入口点程序（若有） |
| `Env` | 默认环境变量 |
| `WorkingDir` | 默认工作目录 |
| `ExposedPorts` | 暴露端口 |
| `Volumes` | 默认挂载卷 |

* * *