```
docker build -t alpine-figlet-from-dockerfile .
```

`docker build` 是 Docker CLI 中用于**构建镜像（Image）**的命令。
根据指定的 **上下文（context）路径** 和 Dockerfile 的内容，自动执行构建流程，逐层创建镜像。

- Docker 会将 `.` 目录作为上下文发送给 Docker 守护进程（`dockerd`）。
    
- 解析其中的 `Dockerfile`，按照其内容指令（如 `FROM`, `RUN`, `COPY`, `CMD` 等）逐层构建镜像。
    
- 每条指令对应镜像的一层（layer），具备缓存能力。

`-t alpine-figlet-from-dockerfile`
- `alpine-figlet-from-dockerfile` 是构建后的镜像名称（不含版本标签时默认是 `latest`）。