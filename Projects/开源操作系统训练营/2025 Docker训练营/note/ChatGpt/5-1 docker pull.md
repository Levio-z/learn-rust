```
docker pull alpine
➜  /workspace git:(main) docker pull alpine
Using default tag: latest
latest: Pulling from library/alpine
fe07684b16b8: Pull complete 
Digest: sha256:8a1f59ffb675680d47db6337b49d22281a139e9d709335b492be023728e11715
Status: Downloaded newer image for alpine:latest
docker.io/library/alpine:latest
```
- docker pull alpine
	- 含义：从默认仓库 Docker Hub 拉取 `alpine` 镜像（Linux 上最小巧的发行版之一）。
- Using default tag: latest
	- 你没有显式指定 tag（例如：`alpine:3.18`），因此默认使用 `:latest` 标签，也就是该镜像当前的默认最新版本。
- latest: Pulling from library/alpine
	- `library/alpine` 是 Docker Hub 上的 **官方 Alpine 镜像**。
	- `library/` 是 Docker Hub 的官方命名空间。
	- 实际上等价于：docker pull docker.io/library/alpine:latest
- fe07684b16b8: Pull complete 
	- 表示这层镜像（Layer ID 为 `fe07684b16b8`）下载并成功完成拉取。
	- Alpine 镜像非常小，往往只有一层。
- Digest: sha256:8a1f59ffb675680d47db6337b49d22281a139e9d709335b492be023728e11715
	- 该镜像的内容摘要（digest），是一个唯一的 SHA256 哈希。
	- 用于校验完整性，保证镜像一致性。
	- 你可以使用它来拉取镜像的固定版本（防止 `latest` 带来意外更新）：
	- docker pull alpine@sha256:8a1f59ffb675680d47db6337b49d22281a139e9d709335b492be023728e11715
- Status: Downloaded newer image for alpine:latest
	- 表示你本地原先没有这个版本（或者有旧版本），所以拉取了新的。
	- 若已存在最新版本，Docker 会提示：`Image is up to date for alpine:latest`
- docker.io/library/alpine:latest
	- 镜像的完整名。
	- `docker.io`：默认的远程仓库 Docker Hub。
	- `library`：官方镜像的命名空间。
	- `alpine`：镜像名。
	- `latest`：标签（tag）。