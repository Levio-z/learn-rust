有两种取得 Git 项目仓库的方法。 第一种是在现有项目或目录下导入所有文件到 Git 中； 第二种是从一个服务器克隆一个现有的 Git 仓库。
#### 在现有目录中初始化仓库
如果你打算使用 Git 来对现有的项目进行管理，你只需要进入该项目目录并输入：
```console
git init
```

### 克隆远程仓库
克隆远程仓库的时候，自定义本地仓库的名字

```console
$ git clone https://github.com/libgit2/libgit2 mylibgit
```