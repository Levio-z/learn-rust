```
docker build -t alpine-figlet-from-dockerfile .
```

`docker build` 是 Docker CLI 中用于**构建镜像（Image）**的命令。
根据指定的 **上下文（context）路径** 和 Dockerfile 的内容，自动执行构建流程，逐层创建镜像。

- `.`
	- Docker 会将 `.` 目录作为上下文发送给 Docker 守护进程（`dockerd`）。
	- `.` 目录指的是**当前终端所在的目录**，也就是你执行 `docker build` 命令时所在的那个目录。
- build
	- 解析其中的 `Dockerfile`，按照其内容指令（如 `FROM`, `RUN`, `COPY`, `CMD` 等）逐层构建镜像。
	- 每条指令对应镜像的一层（layer），具备缓存能力。
`-t alpine-figlet-from-dockerfile`
- `alpine-figlet-from-dockerfile` 是构建后的镜像名称（不含版本标签时默认是 `latest`）。
- 相当于`alpine-figlet-from-dockerfile:latest`

### 示例-2
```
docker build -t golang-demo-single -f golang_sample/Dockerfile.single golang_sample/

```

| 参数                                   | 含义                                                      |
| ------------------------------------ | ------------------------------------------------------- |
| `docker build`                       | 构建镜像的主命令                                                |
| `-t golang-demo-single`              | 给构建的镜像打标签（tag），这里将镜像命名为 `golang-demo-single`            |
| `-f golang_sample/Dockerfile.single` | 指定 Dockerfile 的路径为 `golang_sample/Dockerfile.single`    |
| `golang_sample/`                     | 构建上下文（build context）目录。Dockerfile 中 `COPY` 等操作会相对于此路径执行 |
### 实践
```
docker run -d -p 8888:8888  jupyter-sample
```