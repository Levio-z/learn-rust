>https://git-scm.com/book/zh/v2/Git-%e5%86%85%e9%83%a8%e5%8e%9f%e7%90%86-Git-%e5%af%b9%e8%b1%a1

### .git/objects 目录详解

`.git/objects` 目录是 Git 版本控制系统中非常核心的组件，用于存储 Git 中的所有数据对象。从代码中可以看到，它是在执行 `git init` 命令时创建的一个关键目录。

#### 主要功能

`.git/objects` 目录的主要作用是：

1. **存储所有版本的数据内容**：包括文件内容、目录结构、提交记录等
2. **实现 Git 的内容寻址存储系统**：每个对象都由其内容的 SHA-1 哈希值唯一标识
3. **支持数据压缩和高效存储**：通过各种对象类型和打包机制优化存储空间

#### 目录结构

在当前的测试仓库中，`.git/objects` 目录包含两个子目录：
- **info/**：通常包含对象的额外信息，比如打包文件的索引信息
- **pack/**：用于存储打包后的对象文件，Git 会定期将多个松散对象打包成一个文件以节省空间

#### 工作原理

Git 中有四种基本对象类型，都会存储在这个目录中：

1. **blob 对象**：存储文件内容
2. **tree 对象**：存储目录结构和文件元数据
3. **commit 对象**：存储提交信息，包括作者、时间、提交消息和指向树对象的指针
4. **tag 对象**：存储标签信息

当你创建或修改文件并执行 `git add` 命令时，Git 会计算文件内容的哈希值，并将内容压缩后存储在 `.git/objects` 目录下，文件名就是哈希值。目录通常会按照哈希值的前两位字符创建子目录，后38位作为文件名，这样可以提高文件系统的访问效率。
#### object存储格式
```
blob <size>\0<content>
```
- `<size>` is the size of the content (in bytes)
- `\0` is a null byte
- `<content>` is the actual content of the file

```console
 header = "blob #{content.length}\0"
```
Git 会将上述头部信息和原始数据拼接起来，并计算出这条新内容的 SHA-1 校验和
```console
bd9dbf5aae1a3862dd1526723246b20206e5fc37
```
Git 会通过 zlib 压缩这条新内容

经由 zlib 压缩的内容写入磁盘上的某个对象。 要先确定待写入对象的路径（SHA-1 值的前两个字符作为子目录名称，后 38 个字符则作为子目录内文件的名称）。
- 创建对应目录
- 写入文件



### .git/objects基本操作
Git 是一个内容寻址文件系统，听起来很酷。但这是什么意思呢？ 这意味着，Git 的核心部分是一个简单的键值对数据库（key-value data store）。 你可以向 Git 仓库中插入任意类型的内容，它会返回一个唯一的键，通过该键可以在任意时刻再次取回该内容。

可以通过底层命令 `git hash-object` 来演示上述效果——该命令可将任意数据保存于 `.git/objects` 目录（即 **对象数据库**），并返回指向该数据对象的唯一的键。

首先，我们需要初始化一个新的 Git 版本库，并确认 `objects` 目录为空：


```console
$ git init test
Initialized empty Git repository in /tmp/test/.git/
$ cd test
$ find .git/objects
.git/objects
.git/objects/info
.git/objects/pack
$ find .git/objects -type f
```
可以看到 Git 对 `objects` 目录进行了初始化，并创建了 `pack` 和 `info` 子目录，但均为空。 接着，我们用 `git hash-object` 创建一个新的数据对象并将它手动存入你的新 Git 数据库中：
#### `git hash-object`详解

```console
$ echo 'test content' | git hash-object -w --stdin
d670460b4b4aece5915caf5c68d12f560a9fe3e4
```

`git hash-object` 会接受你传给它的东西，而它只会返回可以存储在 Git 仓库中的唯一键
 `-w` 选项会指示该命令不要只返回键，还要将该对象写入数据库中。
 `--stdin` 选项则指示该命令从标准输入读取内容；若不指定此选项，则须在命令尾部给出待存储文件的路径。

还可以应用于文件

```console
git hash-object -w test.txt
```

#### 查看.git/objects文件
```console
$ find .git/objects -type f
.git/objects/d6/70460b4b4aece5915caf5c68d12f560a9fe3e4
```
SHA-1 校验和为文件命名
- 一个文件对应一条内容， 以该内容加上特定头部信息一起的 SHA-1 校验和为文件命名。 校验和的前两个字符用于命名子目录，余下的 38 个字符则用作文件名。
#### cat-file 查看文件内容
我们用 `git cat-file` 查看一下该对象的内容：
```console
$ git cat-file -p bd9dbf5aae1a3862dd1526723246b20206e5fc37
what is up, doc?
```
所有的 Git 对象均以这种方式存储，区别仅在于类型标识——另两种对象类型的头部信息以字符串“commit”或“tree”开头，而不是“blob”。 另外，虽然数据对象的内容几乎可以是任何东西，但提交对象和树对象的内容却有各自固定的格式。
#### 多版本案例
```console
$ echo 'version 1' > test.txt
$ git hash-object -w test.txt
83baae61804e65cc73a7201a7252750c76066a30
```

接着，向文件里写入新内容，并再次将其存入数据库：

```console
$ echo 'version 2' > test.txt
$ git hash-object -w test.txt
1f7a7a472abf3dd9643fd615f6da379c4acb3e3a
```

对象数据库记录下了该文件的两个不同版本，当然之前我们存入的第一条内容也还在：

```console
$ find .git/objects -type f
.git/objects/1f/7a7a472abf3dd9643fd615f6da379c4acb3e3a
.git/objects/83/baae61804e65cc73a7201a7252750c76066a30
.git/objects/d6/70460b4b4aece5915caf5c68d12f560a9fe3e4
```

现在可以在删掉 `test.txt` 的本地副本，然后用 Git 从对象数据库中取回它的第一个版本：

```console
$ git cat-file -p 83baae61804e65cc73a7201a7252750c76066a30 > test.txt
$ cat test.txt
version 1
```

或者第二个版本：

```console
$ git cat-file -p 1f7a7a472abf3dd9643fd615f6da379c4acb3e3a > test.txt
$ cat test.txt
version 2
```
然而，记住文件的每一个版本所对应的 SHA-1 值并不现实；另一个问题是，在这个（简单的版本控制）系统中，文件名并没有被保存——我们仅保存了文件的内容。 上述类型的对象我们称之为 **数据对象（blob object）**。 利用 `git cat-file -t` 命令，可以让 Git 告诉我们其内部存储的任何对象类型，只要给定该对象的 SHA-1 值：
```console
$ git cat-file -t 1f7a7a472abf3dd9643fd615f6da379c4acb3e3a
blob
```
### 树对象

接下来要探讨的 Git 对象类型是树对象（tree object），它能解决文件名保存的问题，也允许我们将多个文件组织到一起。 Git 以一种类似于 UNIX 文件系统的方式存储内容，但作了些许简化。 所有内容均**以树对象和数据对象的形式存储，其中树对象对应了 UNIX 中的目录项，数据对象则大致上对应了 inodes 或文件内容**。 一个树对象包含了一条或多条树对象记录（tree entry），每条记录含有一个指向数据对象或者子树对象的 SHA-1 指针，以及相应的模式、类型、文件名信息。 例如，某项目当前对应的最新树对象可能是这样的：

```console
$ git cat-file -p master^{tree}
100644 blob a906cb2a4a904a152e80877d4088654daad0c859      README
100644 blob 8f94139338f9404f26296befa88755fc2598c289      Rakefile
040000 tree 99f1a6d12cb4b6f19c8655fca46c3ecf317074e0      lib
```

`master^{tree}` 语法表示 `master` 分支上最新的提交所指向的树对象。 请注意，`lib` 子目录（所对应的那条树对象记录）并不是一个数据对象，而是一个指针，其指向的是另一个树对象：
```console
$ git cat-file -p 99f1a6d12cb4b6f19c8655fca46c3ecf317074e0
100644 blob 47c6340d6459e05787f644c2447d2595f5d3a54b      simplegit.rb
```

|   |
|---|
|你可能会在某些 shell 中使用 `master^{tree}` 语法时遇到错误。<br><br>在 Windows 的 CMD 中，字符 `^` 被用于转义，因此你必须双写它以避免出现问题：`git cat-file -p master^^{tree}`。 在 PowerShell 中使用字符 {} 时则必须用引号引起来，以此来避免参数解析错误：`git cat-file -p 'master^{tree}'`。<br><br>在 ZSH 中，字符 `^` 被用在通配模式（globbing）中，因此你必须将整个表达式用引号引起来：`git cat-file -p "master^{tree}"`。|

![](asserts/Pasted%20image%2020250913072655.png)

Figure 148. 简化版的 Git 数据模型。
#### git update-index
**索引（index/staging area）** 就是 Git 记录「工作区文件准备提交的状态」的地方。
当你执行 `git write-tree` 时，Git 会把索引中的文件条目序列化为一个 **Tree 对象**，写入 `.git/objects/` 中。
Tree 对象中的每一条记录对应索引里的一个条目。

```console
$ git update-index --add --cacheinfo 100644 \
  83baae61804e65cc73a7201a7252750c76066a30 test.txt
```
 `git update-index` **暂存一些文件来创建一个暂存**区
- `--add`：告诉 Git 要往索引里新增一个条目（哪怕它现在不在工作区）。
- `--cacheinfo <mode> <sha1> <path>`：直接用 `<mode>`、`<sha1>`、`<path>` 添加一个索引项，而不依赖工作区的实际文件。
    - `mode`：Git 文件模式（100644 普通文件，100755 可执行，120000 符号链接）。
    - `sha1`：Blob 对象的 ID（即 `git hash-object` 得到的内容哈希）。
    - `path`：文件名，比如 `test.txt`。
>“索引中有个 `test.txt`，它的内容是 `sha1=83baae...` 对应的 blob，对应模式是 `100644`”。


你可以轻松创建自己的树对象。 通常，Git **根据某一时刻暂存区（即 index 区域，下同）所表示的状态创建并记录一个对应的树对象， 如此重复便可依次记录（某个时间段内）一系列的树对象**。 因此，为创建一个树对象，首先需要通过**暂存一些文件来创建一个暂存区**。 可以通过底层命令 `git update-index` 为一个单独文件——我们的 test.txt 文件的首个版本——创建一个暂存区。 利用该命令，可以把 `test.txt` 文件的首个版本人为地加入一个新的暂存区。 

文件格式：
- **100644** → 普通文件（非可执行，读写权限受限）。
- **100755** → 可执行文件（可运行脚本、二进制）。
- **120000** → 符号链接（symlink，内容是目标路径）。
- **040000** → 目录（指向另一个 Tree 对象）。
- **160000** → Gitlink（子模块，指向一个 commit ID）。

这说明 **Tree 对象是一个目录级别的快照**，里面记录了「文件名、文件模式、指向的对象（Blob/Tree/Commit）」。
####  `git write-tree`
**将暂存区内容写入一个树对象**
现在，可以通过 `git write-tree` 命令**将暂存区内容写入一个树对象**。 此处无需指定 `-w` 选项——如果某个树对象此前并不存在的话，当调用此命令时， 它会根据当前暂存区状态自动创建一个新的树对象：

```console
$ git write-tree
d8329fc1cc938780ffdd9f94e0d364e0ea74f579
$ git cat-file -p d8329fc1cc938780ffdd9f94e0d364e0ea74f579
100644 blob 83baae61804e65cc73a7201a7252750c76066a30      test.txt
```
不妨用之前见过的 `git cat-file` 命令验证一下它确实是一个树对象：
```console
$ git cat-file -t d8329fc1cc938780ffdd9f94e0d364e0ea74f579
tree
```
接着我们来创建一个新的树对象，它包括 `test.txt` 文件的第二个版本，以及一个新的文件：
```console
$ echo 'new file' > new.txt
$ git update-index --add --cacheinfo 100644 \
  1f7a7a472abf3dd9643fd615f6da379c4acb3e3a test.txt
$ git update-index --add new.txt
```
- git update-index --add new.txt
	- Git 为 `new.txt` 生成 Blob 对象
	- 更新索引条目：(mode, sha1, path)
暂存区现在包含了 `test.txt` 文件的新版本，和一个新文件：`new.txt`。 记录下这个目录树（将当前暂存区的状态记录为一个树对象），然后观察它的结构：
```console
$ git write-tree
0155eb4229851634a0f03eb265b69f5a2d56f341
$ git cat-file -p 0155eb4229851634a0f03eb265b69f5a2d56f341
100644 blob fa49b077972391ad58037050f2a75f74e3671e92      new.txt
100644 blob 1f7a7a472abf3dd9643fd615f6da379c4acb3e3a      test.txt
```
我们注意到，新的树对象包含两条文件记录，同时 test.txt 的 SHA-1 值（`1f7a7a`）是先前值的“第二版”。 只是为了好玩：你可以将第一个树对象加入第二个树对象，使其成为新的树对象的一个子目录。 
#### `git read-tree`
通过调用 `git read-tree` 命令，可以把树对象读入暂存区。 本例中，可以通过对该命令指定 `--prefix` 选项，将一个已有的树对象作为子树读入暂存区：

```console
$ git read-tree --prefix=bak d8329fc1cc938780ffdd9f94e0d364e0ea74f579
$ git write-tree
3c4e9cd789d88d8d89c1073707c3585e41b0e614
$ git cat-file -p 3c4e9cd789d88d8d89c1073707c3585e41b0e614
040000 tree d8329fc1cc938780ffdd9f94e0d364e0ea74f579      bak
100644 blob fa49b077972391ad58037050f2a75f74e3671e92      new.txt
100644 blob 1f7a7a472abf3dd9643fd615f6da379c4acb3e3a      test.txt
```

如果基于这个新的树对象创建一个工作目录，你会发现工作目录的根目录包含两个文件以及一个名为 `bak` 的子目录，该子目录包含 `test.txt` 文件的第一个版本。 可以认为 Git 内部存储着的用于表示上述结构的数据是这样的：

![](asserts/Pasted%20image%2020250913073330.png)

Figure 149. 当前 Git 的数据内容结构。
#### 存储格式
- 指向 blob 或树对象的 SHA-1 哈希
- 文件/目录的名称
- 文件/目录的模式
	- `100644` (regular file)  
	- `100755` (executable file)  
	- `120000` (symbolic link)  
	- 对于目录，值为 `040000`

格式：

```
tree <size>\0
<mode> <name>\0<20_byte_sha>
```
```
blob <size>\0<content>
```
案例：
```
```
```
tree [content size]\0[Entries having references to other trees and blobs]
```


### 提交对象
如果你做完了以上所有操作，那么现在就有了三个树对象，分别代表我们想要跟踪的不同项目快照。 然而问题依旧：**若想重用这些快照，你必须记住所有三个 SHA-1 哈希值。 并且，你也完全不知道是谁保存了这些快照，在什么时刻保存的，以及为什么保存这些快照**。 而以上这些，正是提交对象（commit object）能为你保存的基本信息。
#### `commit-tree`
可以通过调用 `commit-tree` 命令创建一个提交对象，为此需要指定一个树对象的 SHA-1 值，以及该提交的父提交对象（如果有的话）。 我们从之前创建的第一个树对象开始：

```console
$ echo 'first commit' | git commit-tree d8329f
fdf4fc3344e67ab068f836878b6c4951e3b15f3d
```
由于创建时间和作者数据不同，你现在会得到一个不同的散列值。 请将本章后续内容中的提交和标签的散列值替换为你自己的校验和。 现在可以通过 `git cat-file` 命令查看这个新提交对象：
```console
$ git cat-file -p fdf4fc3
tree d8329fc1cc938780ffdd9f94e0d364e0ea74f579
author Scott Chacon <schacon@gmail.com> 1243040974 -0700
committer Scott Chacon <schacon@gmail.com> 1243040974 -0700

first commit
```

提交对象的格式很简单：
- 它先指定一个顶层树对象，代表当前项目快照； 
- 然后是可能存在的父提交（前面描述的提交对象并不存在任何父提交）； 
- 之后是作者/提交者信息（依据你的 `user.name` 和 `user.email` 配置来设定，外加一个时间戳）； 留空一行，最后是提交注释。
#### 提交历史
接着，我们将创建另两个提交对象，它们分别引用各自的上一个提交（作为其父提交对象）：

```console
$ echo 'second commit' | git commit-tree 0155eb -p fdf4fc3
cac0cab538b970a37ea1e769cbbde608743bc96d
$ echo 'third commit'  | git commit-tree 3c4e9c -p cac0cab
1a410efbd13591db07496601ebc7a059dd55cfe9
```
这三个提交对象分别指向之前创建的三个树对象快照中的一个。 现在，如果对最后一个提交的 SHA-1 值运行 `git log` 命令，会出乎意料的发现，你已有一个货真价实的、可由 `git log` 查看的 Git 提交历史了：
```console
$ git log --stat 1a410e
commit 1a410efbd13591db07496601ebc7a059dd55cfe9
Author: Scott Chacon <schacon@gmail.com>
Date:   Fri May 22 18:15:24 2009 -0700

	third commit

 bak/test.txt | 1 +
 1 file changed, 1 insertion(+)

commit cac0cab538b970a37ea1e769cbbde608743bc96d
Author: Scott Chacon <schacon@gmail.com>
Date:   Fri May 22 18:14:29 2009 -0700

	second commit

 new.txt  | 1 +
 test.txt | 2 +-
 2 files changed, 2 insertions(+), 1 deletion(-)

commit fdf4fc3344e67ab068f836878b6c4951e3b15f3d
Author: Scott Chacon <schacon@gmail.com>
Date:   Fri May 22 18:09:34 2009 -0700

    first commit

 test.txt | 1 +
 1 file changed, 1 insertion(+)
```

太神奇了： 就在刚才，你没有借助任何上层命令，仅凭几个底层操作便完成了一个 Git 提交历史的创建。 这就是每次我们运行 `git add` 和 `git commit` 命令时，Git 所做的工作实质就是将被改写的文件保存为数据对象， 更新暂存区，记录树对象，最后创建一个指明了顶层树对象和父提交的提交对象。 这三种主要的 Git 对象——数据对象、树对象、提交对象——最初均以单独文件的形式保存在 `.git/objects` 目录下。 下面列出了目前示例目录内的所有对象，辅以各自所保存内容的注释：
```console
$ find .git/objects -type f
.git/objects/01/55eb4229851634a0f03eb265b69f5a2d56f341 # tree 2
.git/objects/1a/410efbd13591db07496601ebc7a059dd55cfe9 # commit 3
.git/objects/1f/7a7a472abf3dd9643fd615f6da379c4acb3e3a # test.txt v2
.git/objects/3c/4e9cd789d88d8d89c1073707c3585e41b0e614 # tree 3
.git/objects/83/baae61804e65cc73a7201a7252750c76066a30 # test.txt v1
.git/objects/ca/c0cab538b970a37ea1e769cbbde608743bc96d # commit 2
.git/objects/d6/70460b4b4aece5915caf5c68d12f560a9fe3e4 # 'test content'
.git/objects/d8/329fc1cc938780ffdd9f94e0d364e0ea74f579 # tree 1
.git/objects/fa/49b077972391ad58037050f2a75f74e3671e92 # new.txt
.git/objects/fd/f4fc3344e67ab068f836878b6c4951e3b15f3d # commit 1
```

#### 对象关系图
如果跟踪所有的内部指针，将得到一个类似下面的对象关系图：
![](asserts/Pasted%20image%2020250913073811.png)



