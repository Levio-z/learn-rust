```
COPY go.mod .
COPY main.go .
```

**定义与作用：**  
将宿主机当前目录下的 `go.mod` 和 `main.go` 拷贝进镜像的 `/app/` 目录中。

**原理：**  
Docker 会将宿主机文件打包进构建上下文中，并复制到目标路径。

**扩展知识：**

- 使用 `.dockerignore` 排除不必要的文件（如 `.git`, `vendor`, `*.log` 等）。
    
- 先复制 `go.mod` 再 `main.go` 是为了优化构建缓存：`go mod tidy` 只在 `go.mod` 变化时重新运行。

### 多阶段构建
```
# 从 builder 阶段复制编译好的二进制文件

COPY --from=builder /app/server .
```