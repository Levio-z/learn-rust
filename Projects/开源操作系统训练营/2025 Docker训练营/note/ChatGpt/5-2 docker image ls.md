```shell
➜  /workspace git:(main) docker image ls              
REPOSITORY   TAG       IMAGE ID       CREATED       SIZE
alpine       latest    cea2ff433c61   5 weeks ago   8.3MB
```
显示本地已存在的 Docker 镜像列表。我们来逐列解释你看到的输出

|列名|含义|
|---|---|
|**REPOSITORY**|镜像的仓库名，例如 `alpine`。它可能来源于 Docker Hub，也可能来自私有仓库。|
|**TAG**|镜像标签，如 `latest`、`3.18` 等，用于标识同一镜像的不同版本。|
|**IMAGE ID**|镜像的唯一标识符（SHA 截断形式），用于引用镜像（如在 `docker run`、`docker rmi` 中使用）。|
|**CREATED**|镜像的构建时间，即创建这个镜像的时间戳（不是你 pull 下来的时间）。|
|**SIZE**|镜像的总大小，占据磁盘的空间。例如 Alpine 镜像非常小，只有 8.3MB。|