---
tags:
  - permanent
---
## 1. 核心观点  

#### 定义

**客户端发送want-have请求下载packfile数据**

在完成引用和功能发现后，客户端可以通过发送 flush-pkt 包来终止连接，告知服务器现在可以优雅地终止并断开连接，此时客户端不再需要任何数据包。这可以通过 ls-remote 命令实现，也可以在客户端已是最新版本的情况下实现。

否则，程序将进入协商阶段。在此阶段，客户端和服务器将通过告知服务器所需的对象、浅层对象（如有）以及最大提交深度（如有）来确定传输所需的最小打包文件。客户端还会发送一个列表，列出它希望生效的功能，这些功能基于服务器在第一_行_需求中声明的功能。


## 2. 背景/出处  
- 来源：https://git-scm.com/docs/gitprotocol-pack
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  

### 格式
```
  upload-request    =  want-list
		       *shallow-line
		       *1depth-request
		       [filter-request]
		       flush-pkt

  want-list         =  first-want
		       *additional-want

  shallow-line      =  PKT-LINE("shallow" SP obj-id)

  depth-request     =  PKT-LINE("deepen" SP depth) /
		       PKT-LINE("deepen-since" SP timestamp) /
		       PKT-LINE("deepen-not" SP ref)

  first-want        =  PKT-LINE("want" SP obj-id SP capability-list)
  additional-want   =  PKT-LINE("want" SP obj-id)

  depth             =  1*DIGIT

  filter-request    =  PKT-LINE("filter" SP filter-spec)
```

### 案例：生成 pkt-line 请求体（want + done）

**请求体**

```
0032want 9671f5a72cac8b4c379b1c35a6af6d10611d620f
00000009done
```
Git 的 pkt-line 不要求你发送能力列表（capabilities），最小实现只要 want + done 即可启动 packfile 输出。

**发送请求并把返回内容存成 packfile**

```shell

curl -v \

  -o packfile.bin1 \

  -X POST \

  -H "Content-Type: application/x-git-upload-pack-request" \

  --data-binary $'0032want 9671f5a72cac8b4c379b1c35a6af6d10611d620f\n00000009done\n' \

  "https://github.com/learn-rust-projects/build-your-own-git/git-upload-pack"

```

--data-binary：发送二进制内容，不做任何转义/编码。
-o packfile.bin： 把服务器返回的 packfile 保存为本地文件。




   把服务器返回的 packfile 保存为本地文件。
## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：
	- 【ref】[智能协议测试](智能协议测试.md)
	- [Git-URL Format  URL 格式](Git-URL%20Format%20%20URL%20格式.md)
	- [Git-gitprotocol-pack - pkt-line 格式](Git-gitprotocol-pack%20-%20pkt-line%20格式.md)
## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
- [ ] 深入阅读 xxx  
- [ ] 验证这个观点的边界条件  
