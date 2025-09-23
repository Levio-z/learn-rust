```console
echo 1a410efbd13591db07496601ebc7a059dd55cfe9 > .git/refs/heads/master
```
就是将一个master文件，里面存放最新提交的哈希值

![](asserts/Pasted%20image%2020250911143929.png)


### Git引用

```console
git update-ref refs/heads/master 1a410efbd13591db07496601ebc7a059dd55cfe9
```
回溯记录

```console
git log --pretty=oneline test
```

### HEAD
HEAD 文件通常是一个符号引用（symbolic reference），指向目前所在的分支。 所谓符号引用，表示它是一个指向其他引用的指针。

如果执行 `git checkout test`，Git 会像这样更新 HEAD 文件：
```console
$ cat .git/HEAD
ref: refs/heads/test
```
### git commit
当我们执行 `git commit` 时，该命令会创建一个提交对象，**并用 HEAD 文件中那个引用所指向的 SHA-1 值设置其父提交字段**。将 `HEAD` 指向的引用（如 `refs/heads/main`）更新为这个新提交对象的 SHA-1。
- 假设你在 `main` 分支，Git 会更新 `.git/refs/heads/main` 文件，把里面的 SHA-1 替换为新提交对象的哈希。
- 这样，`HEAD` → `refs/heads/main` → 新提交对象。

### 标签引用
一个永不移动的分支引用——永远指向同一个提交对象

```console
git update-ref refs/tags/v1.0 cac0cab538b970a37ea1e769cbbde608743bc96d
```
### 一个附注标签则更复杂一些

```console
$ git tag -a v1.1 1a410efbd13591db07496601ebc7a059dd55cfe9 -m 'test tag'
```
生成一个标签对象
```console
$ git cat-file -p 9585191f37f7b0fb9444f35a9bf50de191beadc2
object 1a410efbd13591db07496601ebc7a059dd55cfe9
type commit
tag v1.1
tagger Scott Chacon <schacon@gmail.com> Sat May 23 16:48:58 2009 -0700

test tag
```
标签可以打在任何git对象上，就可以方便的来存和查看对象
```
git tag hello-blob 3b18e512dba79e4c8300dd08aeb37f8e728b8dad

git cat-file -t hello-blob
blob

git cat-file -p hello-blob
hello world
```