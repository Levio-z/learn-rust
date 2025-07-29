##### docker version 
查看版本信息
```
Client:
 Version:           26.1.3
 API version:       1.45
 Go version:        go1.21.10
 Git commit:        b72abbb
 Built:             Thu May 16 08:32:30 2024
 OS/Arch:           linux/amd64
 Context:           default

Server: Docker Engine - Community
 Engine:
  Version:          26.1.3
  API version:      1.45 (minimum version 1.24)
  Go version:       go1.21.10
  Git commit:       8e96db1
  Built:            Thu May 16 08:33:58 2024
  OS/Arch:          linux/amd64
  Experimental:     false
 containerd:
  Version:          v1.7.15
  GitCommit:        926c9586fe4a6236699318391cd44976a98e31f1
 runc:
  Version:          1.1.12
  GitCommit:        v1.1.12-0-g51d5e94
 docker-init:
  Version:          0.19.0
  GitCommit:        de40ad0
```

| 字段              | 说明                                                          |
| --------------- | ----------------------------------------------------------- |
| **Version**     | Docker 客户端版本（CLI 版本），目前是 `26.1.3`，属于最新稳定版之一。                |
| **API version** | Docker 使用的 REST API 版本，`1.45`，这与客户端和守护进程通信时的兼容性有关。          |
| **Go version**  | 编译 Docker 时所使用的 Go 语言版本（影响构建效率与安全）。                         |
| **Git commit**  | 对应构建 Docker 的源代码提交哈希，可用于追溯源码。                               |
| **Built**       | 构建时间。                                                       |
| **OS/Arch**     | 客户端运行平台是 `linux/amd64`。                                     |
| **Context**     | 当前使用的 `docker context`（用于连接不同远程 Docker 守护进程），默认是 `default`。 |
Server: Docker Engine - Community
- 这是社区版本的 Docker 引擎（非企业版）。

| 组件              | 功能                                   |
| --------------- | ------------------------------------ |
| **containerd**  | 高级容器运行时，管理镜像、容器生命周期，是 Docker 的核心子模块。 |
| **runc**        | 最底层的 OCI 容器运行器，负责具体的容器启动/销毁等低层操作。    |
| **docker-init** | 容器内的初始化程序，PID=1，用于处理僵尸进程、信号代理等。      |
