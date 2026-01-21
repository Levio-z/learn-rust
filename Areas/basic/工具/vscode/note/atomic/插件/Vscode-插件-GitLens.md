---
tags:
  - note
---
## 1. 核心观点  
### Ⅰ. 概念层


### Ⅱ. 应用层
- 编辑器内的 **Blame 注释**（查看每行是谁改的）
	- 右键
		- copy link to github 挺好用的
	- 查看当前版本和之前版本的比较
	- 查看单个文件的修改

### Ⅲ. 实现层


### **IV**.原理层


## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  
### 一、概述：GitLens 是什么

GitLens 是由 **GitKraken** 团队开发和维护的 **Visual Studio Code 开源插件**，旨在增强 VS Code 内置的 Git 功能。  
它能让你更高效地理解、编写和审查代码。

核心功能包括：

- 编辑器内的 **Blame 注释**（查看每行是谁改的）
- **Hover 信息**（悬停显示作者与提交信息）
- **CodeLens**（在函数或模块上方显示作者与修改信息）
- 高级的 **提交图表（Commit Graph）**、**文件历史**、**交互式变基编辑器（Interactive Rebase Editor）**
- **AI 辅助提交信息生成、变更解释、PR 说明编写等功能**
    
---

### 二、版本与授权模式

#### 1. GitLens Community（社区版）

免费功能包括：

- 在文件或代码块顶部显示作者信息
- 实时查看代码历史与修改来源
- 跟踪文件的演变（Revision Navigation）
- Inline Blame、Hover、CodeLens 等核心功能

#### 2. GitLens Pro（专业版）

付费解锁更强功能：

- **PR 审查加速**：在 VS Code 内直接管理 Pull Request
    
- **Commit Graph**：可视化提交历史、支持 rebase、merge、搜索、筛选
    
- **Launchpad 集成**：与 GitHub/GitLab/Bitbucket 无缝对接
    
- **Cloud Patches / Code Suggest**：无需提交即可分享修改
    
- **AI 功能**：自动生成 Commit、PR、ChangeLog
    

> 公共仓库部分 Pro 功能可免费使用。预览功能可能未来转为 Pro 特性。

---

### 三、核心功能详解

#### 1. 代码历史与责任追踪

- **Inline Blame**：在每行代码末尾显示最后修改者与提交时间
- **Status Bar Blame**：在状态栏显示当前行作者与时间
- **Hover 信息**：悬停即可查看详细提交记录与相关操作
- **File Annotations**：为整个文件渲染作者热力图、修改历史、变化频度等

#### 2. Commit Graph（提交图）

- 可视化仓库结构、分支关系、合并与提交历史
    
- 支持搜索作者、文件、提交信息等
    
- 一键执行 rebase / merge / revert 等操作
    

#### 3. Revision Navigation（历史导航）

- 快速在文件历史版本间跳转
    
- 查看文件或单行的修改演变过程
    

#### 4. Launchpad（PR 中枢）

- 将所有 GitHub/GitLab PR 集中在一个视图中管理
    
- 可直接创建、审查、合并、标记优先级
    
- 与 Worktrees 联动，可在多分支上并行工作
    

#### 5. Worktrees（多工作树）

- 在多个分支间同时工作，无需 stash 或切换分支
    
- 支持在新窗口独立打开 PR 进行审查
    

#### 6. Cloud Patches 与 Code Suggest

- **Cloud Patches**：可私密地分享未提交修改，避免污染仓库
    
- **Code Suggest**：像 Google Docs 一样直接对文件建议修改，而不局限于 PR 改动行
    

#### 7. AI 加速功能（预览）

- 自动生成提交信息、PR 描述、变更日志
    
- AI 解释提交意图
    
- 支持 GitHub Copilot、OpenAI、Anthropic、DeepSeek、Gemini 等 API
    

---

### 四、辅助视图与命令

#### 1. Inspect 视图

提供代码的上下文信息：

- 查看提交或暂存（stash）详情
    
- 按行历史（Line History）或文件历史（File History）浏览
    
- 可视化文件变更时间线（Visual File History Pro）
    

#### 2. 侧边栏（Side Bar Views）

模块化视图：

- **Commits / Branches / Tags / Stashes / Remotes**
    
- **Contributors**：查看贡献者排名与贡献量
    
- **Repositories**：多仓库统一管理
    

#### 3. Git 命令面板

无需记忆命令行，通过交互式面板执行：

- 查看提交历史
    
- 文件对比
    
- 分支操作
    
- 暂存与恢复
    

---

### 五、集成与扩展

GitLens 可自动识别并链接外部服务：

- **GitHub / GitHub Enterprise / GitLab / Bitbucket / Jira / Gitea / Gerrit / Azure DevOps**
    
- 支持自动识别 PR、Issue、用户头像
    
- 可自定义自动链接（autolinks），如将提交消息中的 “JIRA-123” 自动跳转到对应任务
    

---

### 六、社区与支持

- **帮助中心**：GitLens Help Center
    
- **问题反馈**：GitHub Issues
    
- **社区讨论**：GitHub Discussions
    
- **支持服务**：GitKraken Support
    
- **Pro 用户**：享有优先邮件支持与定制培训服务
    

---

### 七、开源与贡献

- 核心代码为 **MIT License**
    
- 目录 `plus/` 下的文件受 **LICENSE.plus** 约束（非开源）
    
- 欢迎通过 Pull Request 贡献代码或文档


## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
- [x] 深入阅读 xxx
- [x] 验证这个观点的边界条件
