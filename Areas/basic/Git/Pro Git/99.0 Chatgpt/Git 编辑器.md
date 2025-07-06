## 🧠 一、Git 编辑器是在什么场景下调用的？

Git 只有在需要你**手动输入多行文本**（如 commit message、rebase 信息、merge 冲突注释等）**而你没有通过参数提供这些内容时**，才会调用默认编辑器。

### 🔧 典型场景包括：

|Git 命令|会调用编辑器的情况说明|
|---|---|
|`git commit`|你**没有**加 `-m "message"`，Git 会打开编辑器让你写 commit message|
|`git rebase -i`|交互式变基，打开编辑器修改操作顺序|
|`git merge`|如果有冲突或需要编辑默认 merge 信息|
|`git tag -a`|如果没加 `-m`，也会打开编辑器输入 tag 注释|

---

## 🧐 二、为什么你从没见过 Git 打开编辑器？

以下是几个可能原因：

### ✅ 1. **你总是用 `-m` 参数提交**

bash

复制编辑

`git commit -m "Initial commit"`

这种情况，Git 根本**不会打开编辑器**。

---

### ✅ 2. **你用的是图形化工具（如 SourceTree / VSCode / GitHub Desktop）**

这些工具不会走终端编辑器流程，而是用自己的 UI 弹窗或面板进行输入。

---

### ✅ 3. **你用的是 VSCode 并且 Git 自动帮你配置好了**

很多现代 Git 安装包会自动设置：

bash

复制编辑

`git config --global core.editor "code --wait"`

所以你可能已经在 VSCode 的集成终端里，**无感知地完成了 Git 编辑行为**。

---

### ✅ 4. **你用的系统是 Windows，默认会弹出 Notepad**

而你可能没注意到有个 Notepad 窗口，其实它就是 Git 正在等你输入的界面。