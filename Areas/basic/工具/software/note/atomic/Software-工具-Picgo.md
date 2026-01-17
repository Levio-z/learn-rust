---
tags:
  - note
---
## 1. 核心观点  
### Ⅰ. 概念层



### Ⅱ. 应用层




### Ⅲ. 实现层

### **IV**.原理层


## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### 创建 GitHub Token

1. 打开 GitHub → 点击右上角头像 → `Settings`
2. 左侧选择 `Developer settings` → `Personal access tokens`
3. 选择 `Tokens (classic)` → `Generate new token`
4. 设置 token 权限（**最小权限即可**）：
    - `repo`（如果仓库是私有的）
    - `public_repo`（如果仓库是公开的）
5. 生成后复制下来（**只显示一次**，PicGo 需要使用）

### **PicGo 配置 GitHub 图床**

打开 PicGo → 设置 → 图床设置 → 选择 **GitHub**，然后填入以下字段：

| 字段名称       | 填写示例                                    | 说明                 |
| ---------- | --------------------------------------- | ------------------ |
| 仓库名（repo）  | `your_username/image-hosting`           | 仓库的完整路径            |
| 分支（branch） | `main` 或 `master`（默认分支）                 | GitHub 默认是 `main`  |
| Token      | `ghp_xxxxxxx...`                        | 你刚生成的 GitHub Token |
| 存储路径       | `img/` 或 `images/` 或留空                  | 图片会上传到这个子目录        |
| 自定义域名      | `https://raw.githubusercontent.com`（可选） | 配合路径组成图床 URL       |
| 自定义路径规则    | 可留空或使用 `${filename}`、`${date}` 等变量      | 自动                 |
来到picgo主页面，随便截张图，点击”剪贴板图片"，命名，点击确定。如果看到上传成功就没问题，如果显示失败，没设置代理。picgo代理需单独设置。
## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  

