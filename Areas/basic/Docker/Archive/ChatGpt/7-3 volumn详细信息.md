
```
➜  /workspace git:(main) docker volume inspect nginx_data
[
    {
        "CreatedAt": "2025-07-24T11:59:54Z",
        "Driver": "local",
        "Labels": null,
        "Mountpoint": "/var/lib/docker/volumes/nginx_data/_data",
        "Name": "nginx_data",
        "Options": null,
        "Scope": "local"
    }
]
```

### 1. `"CreatedAt": "2025-07-24T11:59:54Z"`

- **作用**：表示该 Volume 的创建时间（UTC 格式）。
    
- **意义**：可用于审计、排查问题（例如谁创建了大量 Volume）。
    
- **底层机制**：Docker 在创建 volume 时记录创建时间，存储于 volume metadata 中。
    

---

### 2. `"Driver": "local"`

- **作用**：说明该卷是使用哪个**驱动程序（Driver）**创建的。
    
- **"local" 是默认驱动**，意思是这个卷是存在本地文件系统上的。
    
- **其他驱动示例**：
    
    - `nfs`：挂载 NFS 网络卷
        
    - `azurefile`、`vsphere`、`aws` 等云服务驱动
        
- **扩展**：可使用 `docker plugin` 安装第三方卷插件。
    

---

### 3. `"Labels": null`

- **作用**：Docker 支持为 Volume 添加标签（key-value 对），便于分类、过滤、自动化管理。
    
- **这里为 null**：说明你创建 volume 时没有使用 `--label` 参数。
    
文人
示例（带 label 创建）：

bash

复制编辑

`docker volume create --label env=prod nginx_data`

---

### 4. `"Mountpoint": "/var/lib/docker/volumes/nginx_data/_data"`

- **作用**：这是卷在宿主机上的实际挂载路径（即数据真正存储的位置）。
    
- **关键机制**：
    
    - 当你执行 `docker run -v nginx_data:/usr/share/nginx/html` 时，容器内的 `/usr/share/nginx/html` 会挂载这个目录。
        
    - 宿主机路径 `/var/lib/docker/volumes/nginx_data/_data` 会自动映射成容器内的数据目录。
        

> ✅ 可以直接在宿主机上访问它，但不推荐手动修改（防止绕过 Docker 的数据一致性机制）。

---

### 5. `"Name": "nginx_data"`

- **作用**：这个 volume 的唯一标识符（名称）。
    
- **用户可以自定义名称**，否则 Docker 会随机生成一串 hash 名称（例如 `4a5d6fe8ea...`）。
    

---

### 6. `"Options": null`

- **作用**：卷驱动参数，例如文件系统类型、NFS 地址、只读挂载等。
    
- **此处为 null**：表示未指定特殊挂载选项。
    

示例（NFS）：

bash

复制编辑

`docker volume create \   --driver local \   --opt type=nfs \   --opt o=addr=192.168.1.10,rw \   --opt device=:/path/to/dir \   nginx_nfs`

---

### 7. `"Scope": "local"`

- **作用**：指定卷的作用范围。
    
- **"local"** 表示卷只能在当前宿主机上使用。
    
- 在 Swarm 集群中，还可能出现：
    
    - `"global"`（每个节点都有）
        
    - `"swarm"`（集群共享）