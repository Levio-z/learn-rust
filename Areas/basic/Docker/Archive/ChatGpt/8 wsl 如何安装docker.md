### wsl
```
sudo apt update 
sudo apt install docker.io 
sudo systemctl enable --now docker 
sudo usermod -aG docker $USER 
```

安装 Compose v2 插件（Linux / WSL）
```
mkdir -p ~/.docker/cli-plugins
curl -SL https://github.com/docker/compose/releases/download/v2.24.6/docker-compose-linux-x86_64 -o ~/.docker/cli-plugins/docker-compose
chmod +x ~/.docker/cli-plugins/docker-compose
```
检查当前状态
```
docker version
docker compose version
which docker-compose
```

```
Error response from daemon: Get "https://registry-1.docker.io/v2/": net/http: request canceled while waiting for connection (Client.Timeout exceeded while awaiting headers)
```

### 设置代理
查看docker是否配置代理
```
systemctl show docker | grep -i proxy
```
查看系统代理
```
env | grep -i proxy
```
编辑 systemd 配置：
```
sudo mkdir -p /etc/systemd/system/docker.service.d
sudo nano /etc/systemd/system/docker.service.d/http-proxy.conf
```
填入你的代理地址（示例是本地代理 7890）：
```
[Service]
Environment="HTTP_PROXY=http://127.0.0.1:7890"
Environment="HTTPS_PROXY=http://127.0.0.1:7890"
```
然后重载并重启：
```
sudo systemctl daemon-reexec
sudo systemctl daemon-reload
sudo systemctl restart docker
```
docker镜像拉取测试
```
docker pull hello-world 
```