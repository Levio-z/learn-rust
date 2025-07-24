
### 命令格式
```
docker image ls [OPTIONS] [REPOSITORY[:TAG]]
```
### 示例
```shell
➜  /workspace git:(main) docker image ls              
REPOSITORY   TAG       IMAGE ID       CREATED       SIZE
alpine       latest    cea2ff433c61   5 weeks ago   8.3MB
```
显示本地已存在的 Docker 镜像列表。我们来逐列解释你看到的输出
### 内容解释

|列名|含义|
|---|---|
|**REPOSITORY**|镜像的仓库名，例如 `alpine`。它可能来源于 Docker Hub，也可能来自私有仓库。|
|**TAG**|镜像标签，如 `latest`、`3.18` 等，用于标识同一镜像的不同版本。|
|**IMAGE ID**|镜像的唯一标识符（SHA 截断形式），用于引用镜像（如在 `docker run`、`docker rmi` 中使用）。|
|**CREATED**|镜像的构建时间，即创建这个镜像的时间戳（不是你 pull 下来的时间）。|
|**SIZE**|镜像的总大小，占据磁盘的空间。例如 Alpine 镜像非常小，只有 8.3MB。|
### 常用选项详解

| 选项               | 含义                                        |
| ---------------- | ----------------------------------------- |
| `-a`, `--all`    | 显示所有镜像，包括中间层镜像（intermediate image layers） |
| `--digests`      | 显示镜像内容摘要（Digest）                          |
| `--filter`, `-f` | 使用条件过滤镜像，例如：`dangling=true`               |
| `--format`       | 使用 Go 模板格式化输出结果                           |
| `--no-trunc`     | 显示完整的镜像 ID（不截断）                           |
| `--quiet`, `-q`  | 仅显示镜像 ID（常用于脚本）                           |
| --no-trunc       | 完整显示镜像ID                                  |
### 📌 示例用法

#### 1️⃣ 列出本地镜像（默认）

```bash
docker image ls
```

#### 2️⃣ 显示所有镜像（包括中间层镜像）

```bash
docker image ls -a
```

#### 3️⃣ 显示镜像摘要（Digest）

```bash
docker image ls --digests
```

#### 4️⃣ 按条件过滤 dangling 镜像（未被任何 tag 引用）

```bash
docker image ls -f dangling=true
```

#### 5️⃣ 按镜像名过滤（显示 nginx 镜像）

```bash
docker image ls nginx
```

#### 6️⃣ 自定义输出格式

```bash
docker image ls --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
```

输出示例（自定义格式）：

```nginx
REPOSITORY      TAG         SIZE
nginx           latest      142MB
ubuntu          20.04       72MB
```

#### 7️⃣ 只输出镜像 ID（用于脚本处理）

```bash
docker image ls -q
```
