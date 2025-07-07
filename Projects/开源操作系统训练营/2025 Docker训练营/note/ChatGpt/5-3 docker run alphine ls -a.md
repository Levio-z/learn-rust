运行一个 Alpine 容器并执行 `ls -a` 命令

|部分|含义|
|---|---|
|`docker run`|启动一个容器|
|`alpine`|使用的镜像，表示从本地或远程拉取 `alpine:latest` 镜像|
|`ls -a`|容器启动后要执行的命令：列出所有文件，包括隐藏文件（如 `.profile`）|
