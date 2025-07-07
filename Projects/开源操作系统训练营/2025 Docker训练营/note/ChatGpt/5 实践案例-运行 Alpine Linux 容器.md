Alpine 镜像在企业生产环境中被广泛应用，它是一个极简的 Linux 发行版，

只包含最基本的命令和工具，因此镜像非常小，只有 5MB 左右，并且内置包管理系统 `apk`, 使其成为许多其他镜像的常用起点。

拉取镜像
``` shell

docker pull alpine  #拉取镜像

docker image ls     #查看镜像

```
[5-1 docker pull](5-1%20docker%20pull.md)


运行容器
``` shell

docker run alpine ls -a  #运行容器

docker ps -a             #查看容器

```

交互式运行容器

docker run 命令默认使用镜像中的 Cmd 作为容器的启动命令，Cmd 可通过如下命令来查看。
``` shell

docker inspect alpine --format='{{.Config.Cmd}}'

```
