```
FROM golang:1.23

  

# 定义构建参数

ARG PORT=8080

# 设置环境变量

ENV PORT=${PORT}

	  

WORKDIR /app

  

COPY go.mod .

COPY main.go .

  

ENV CGO_ENABLED=0

ENV GOOS=linux

  

RUN go mod tidy && go build -ldflags="-w -s" -o server .

  

EXPOSE ${PORT}

  

CMD ["./server"]
```

```
FROM golang:1.23

```

**定义与作用：**  
指定基础镜像，这里使用的是官方提供的 `golang` 镜像，版本为 `1.23`。

**原理：**  
这是 Docker 的构建阶段起点。`FROM` 声明告诉 Docker 使用哪个操作系统/语言环境作为构建的基础层。

**使用场景：**  
适用于需要编译或运行 Go 应用的环境。Golang 镜像基于 Debian 或 Alpine，并预装了 `go` 工具链。

**扩展知识：**  
你可以加 `AS builder` 创建多阶段构建，例如：
```dockerfile
FROM golang:1.23 AS builder


以减少最终镜像体积。

```

ARG PORT=8080




**定义与作用：**  
定义构建参数 `PORT`，默认值为 `8080`。这个变量仅在构建阶段可用。

**原理：**  
`ARG` 是构建时变量，可以通过 `--build-arg PORT=9090` 设置，不会保存在镜像运行环境中。

**扩展知识：**
- `ARG` 与 `ENV` 不同，前者不参与最终运行容器的环境变量系统。
- 通常 `ARG` 用于构建逻辑参数控制，例如设置代理、端口、版本等。

---


ENV PORT=${PORT}


**定义与作用：**  
设置环境变量 `PORT`，值为上一步的构建参数 `PORT`。

**原理：**  
`ENV` 是运行时环境变量，会被保存在镜像中，容器启动时可以读取。

**使用场景：**  
在容器内访问端口或配置项（如数据库、路径等）。

---



```
WORKDIR /app
```

**定义与作用：**  
设置当前工作目录为 `/app`。后续的 `COPY` 和 `RUN` 命令都在该目录下执行。

**原理：**  
`WORKDIR` 会自动创建目录（若不存在），并改变当前指令执行目录。

**使用场景：**  
统一源码、构建和运行目录结构，避免路径混乱。

---



```dockerfile
COPY go.mod .
COPY main.go .
```

**定义与作用：**  
将宿主机当前目录下的 `go.mod` 和 `main.go` 拷贝进镜像的 `/app/` 目录中。

**原理：**  
Docker 会将宿主机文件打包进构建上下文中，并复制到目标路径。

**扩展知识：**

-   使用 `.dockerignore` 排除不必要的文件（如 `.git`, `vendor`, `*.log` 等）。
    
-   先复制 `go.mod` 再 `main.go` 是为了优化构建缓存：`go mod tidy` 只在 `go.mod` 变化时重新运行。
    

---

```
ENV CGO\_ENABLED=0  
ENV GOOS=linux

```

**定义与作用：**  
- `CGO_ENABLED=0`：禁用 CGO，使编译结果是纯静态链接（无依赖）。
- `GOOS=linux`：指定构建目标系统为 Linux。

**原理：**  
Go 的交叉编译机制，允许在一个系统上为另一个系统构建二进制。

**使用场景：**
- 生成可移植的 Linux 可执行文件。
- 禁用 CGO 后可在 scratch 或 distroless 镜像中运行（无需 glibc）。

---

```dockerfile
RUN go mod tidy && go build -ldflags="-w -s" -o server .
```

**定义与作用：**

-   `go mod tidy`：清理并同步依赖（添加遗漏，移除多余）。
    
-   `go build`：构建 Go 应用，输出为 `server`。
    
-   `-ldflags="-w -s"`：构建时剥离调试信息以减小体积。

- `-o` 是 Go 编译器 `go build` 的一个参数，用于指定编译生成的输出文件名。

- `server` 是你期望生成的可执行文件名。
    

**原理：**  

构建指令会执行于容器中，产物也保存在镜像中。

**扩展知识：**

-   `-w`：去除 DWARF 调试信息。
    
-   `-s`：去除符号表和调试信息。
    
-   若你需要最小化镜像，下一步可以将二进制复制到一个干净的基础镜像（如 `scratch`）。
    

---



```
EXPOSE ${PORT}
```


**定义与作用：**  
告诉 Docker 镜像默认监听的端口（用于文档和工具），不会实际开放端口。

**原理：**  
不会影响容器运行的行为，更多是**元信息（metadata）**。要真正映射端口需 `docker run -p 8080:8080`。

**扩展知识：**
- 多个端口可以 `EXPOSE 8080 9090`。
- 实际用途是配合编排工具如 Docker Compose、Kubernetes 做端口推理。

为什么说 EXPOSE 是“不会影响容器运行的行为”？

因为：

- 不写 `EXPOSE`，你照样可以用 `-p` 做端口映射，程序照常运行。
    
- 写了 `EXPOSE`，如果不手动映射，外部依然无法访问端口。
---

```

CMD ["./server"]

```

**定义与作用：**  
设置默认容器启动命令为 `./server`。

**原理：**

-   CMD 是**运行阶段默认命令**，若 `docker run` 指定了其它命令，会覆盖它。
    
-   这里使用 JSON array 形式避免 shell 解析问题。
    

**使用场景：**  
设置主程序启动入口，例如 Web 服务、API Server 等。

---

## 总结执行流程逻辑图：

```text
Docker 构建阶段：
 ↓
FROM golang → 安装 go 环境
 ↓
ARG + ENV → 设置编译和运行参数（端口）
 ↓
WORKDIR /app → 切换工作目录
 ↓
COPY go.mod, main.go → 拷贝源代码
 ↓
ENV CGO_ENABLED=0 + GOOS → 设置交叉编译环境
 ↓
RUN go mod tidy + go build → 构建可执行文件 server
 ↓
EXPOSE → 指明容器监听端口（元数据）
 ↓
CMD ["./server"] → 启动服务程序
```

---

如果你想进一步优化这个镜像，比如构建更小的最终镜像，可以引入**多阶段构建**，如：

```dockerfile
FROM golang:1.23 as builder
WORKDIR /app
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-w -s" -o server .

FROM scratch
COPY --from=builder /app/server /server
EXPOSE 8080
CMD ["/server"]
```