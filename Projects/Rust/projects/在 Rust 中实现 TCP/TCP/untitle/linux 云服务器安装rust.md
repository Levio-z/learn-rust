1、导出脚本
`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > rust.sh`
2、打开 rust.sh 脚本
```
9 # If RUSTUP_UPDATE_ROOT is unset or empty, default it. 10 RUSTUP_UPDATE_ROOT="${RUSTUP_UPDATE_ROOT:-https://static.rust-lang.org/rustup}"
```
将 RUSTUP_UPDATE_ROOT 修改为

```ini
RUSTUP_UPDATE_ROOT="http://mirrors.ustc.edu.cn/rust-static/rustup"
```
3、修改环境变量

```bash
export RUSTUP_DIST_SERVER=https://mirrors.tuna.tsinghua.edu.cn/rustup
```

```
source ~/.bashrc
```
这让 rustup-init从国内进行下载rust的组件，提高速度


4、最后执行修改后的rust.sh

```mipsasm
bash rust.sh
```
5、安装git
1. **CentOS / RHEL / 阿里云默认镜像（yum 包管理）**
```rust
sudo yum install -y git
```
2. **Ubuntu / Debian（apt 包管理）**
```
sudo apt update
sudo apt install -y git
```
