## 一、什么是 Volume？

Volume 是 Docker 提供的由 Docker 管理的一块**宿主机文件系统上的目录**，主要用来：

-   **容器之间共享数据**
    
-   **持久化重要数据**
    
-   **避免数据被写进容器层，减小镜像体积，提高性能**
    

Docker Volume 通常存储在：

```bash
/var/lib/docker/volumes/
```

---

## 二、Volume 的使用方式

### 1\. 创建 Volume

```bash
docker volume create my-volume
```

### 2\. 查看所有 Volume

```bash
docker volume ls
```

### 3\. 查看 Volume 详细信息

```bash
docker volume inspect my-volume

```
### 4\. 删除 Volume

```bash
docker volume rm my-volume
```

> ⚠️ 注意：不能删除正在被容器使用的卷。

---

## 三、挂载 Volume 到容器中

### ✅ 推荐方式（具备持久化、隔离等优势）：

```bash
docker run -d \
  --name my-container \
  -v my-volume:/app/data \
  nginx
```

解释：

-   `my-volume` 是已经存在的 Volume（或会自动创建）
-   `/app/data` 是容器内的路径，映射到宿主机上的该 Volume
    

### 查看 Volume 映射关系：

```bash
docker inspect my-container
```

---

## 四、Volume 与 Bind Mount 对比

| 特性     | Volume                    | Bind Mount    |
| ------ | ------------------------- | ------------- |
| 管理方式   | Docker 管理                 | 用户指定路径        |
| 数据存储路径 | `/var/lib/docker/volumes` | 任意宿主机路径       |
| 隔离性    | 高（与宿主机解耦）                 | 低（直接暴露宿主路径）   |
| 安全性    | 更安全                       | 安全性低，权限易错     |
| 推荐使用场景 | 应用数据持久化、容器间共享             | 本地开发调试、配置文件挂载 |

---

## 五、Volume 使用场景

1.  **数据库数据持久化**
    
    ```bash
    docker run -v db_data:/var/lib/mysql mysql
    ```
    
2.  **配置文件挂载**
    
    ```bash
    docker run -v my_config:/etc/nginx nginx
    ```
    
3.  **多容器共享数据**
    
    ```bash
    docker run -v shared_data:/shared busybox
    docker run -v shared_data:/shared alpine
    ```
    
4.  **Docker Compose 中定义卷**
    

```yaml
version: '3'
services:
  db:
    image: mysql
    volumes:
      - db_data:/var/lib/mysql

volumes:
  db_data:
```

---

## 六、补充：匿名卷与命名卷

| 类型 | 用法示例 | 说明 |
| --- | --- | --- |
| 命名卷 | `-v my-volume:/data` | 明确命名，可重复使用 |
| 匿名卷 | `-v /data` | 没有指定卷名，Docker 自动创建 |
| Bind Mount | `-v /host/path:/data` | 使用宿主机指定目录挂载 |

---

## 七、如何查看卷内容？

```bash
docker run --rm -v my-volume:/data busybox ls /data
```

或直接进入宿主机的存储路径：

```bash
cd /var/lib/docker/volumes/my-volume/_data
```

---

## 总结一句话：

> Volume 是 Docker 提供的官方数据持久化机制，**推荐用于生产环境中需要持久保存的数据存储**，比 Bind Mount 更隔离、更安全、更易迁移。

---