>https://blog.meain.io/2023/what-is-in-dot-git/

### git init 后仓库的内容
```
$ tree .git

.git
├── config
├── HEAD
├── hooks
│   └── prepare-commit-msg.msample
├── objects
│   ├── info
│   └── pack
└── refs
    ├── heads
    └── tags


```
- `config` 是一个文本文件，其中包含当前存储库的 Git 配置。如果你查看它，你会看到你的存储库的一些基本设置，比如作者、文件模式等。
- `HEAD` 包含存**储库的当前头**。
	- 根据您设置的“默认”分支，它将是 `refs/heads/master` 或 `refs/heads/main` 或您设置的任何其他分支。正如您可能已经猜到的那样，它指向您可以在下面看到的 `refs/heads` 文件夹，以及一个名为 `master` 的文件，该文件目前不存在。此文件`母版`仅在您进行第一次提交后才会显示。
- `hook`包含可以在 Git 执行任何作之前/之后运行的任何脚本。我[在这里](https://blog.meain.io/2019/making-sure-you-wont-commit-conflict-markers/)写了另一篇博客，详细介绍了 git hooks 的工作原理。
- `objects` 包含 git 对象，**即有关存储库中文件、提交等的数据**。我们将在本博客中深入探讨这一点。
- `refs` 正如我们之前提到的，存储引用（指针）。`refs/heads` 包含指向分支的指针，`refs/tags` 包含指向标记的指针。我们很快就会介绍这些文件的外观。


### 现在我们添加一个文件

现在您已经了解了 `.git` 中的初始文件集是什么，让我们执行第一个操作，向 `.git` 目录添加一些内容。让我们创建一个文件并添加它（我们还没有提交它）。

```bash
echo 'meain.io' > filegit add file
```

这将执行以下操作：
```
--- init       2024-07-02 15:14:00.584674816 +0530
+++ add        2023-07-02 15:13:53.869525054 +0530
@@ -3,7 +3,10 @@
 ├── HEAD
 ├── hooks
 │   └── prepare-commit-msg.msample
+├── index
 ├── objects
+│   ├── 4c
+│   │   └── 5b58f323d7b459664b5d3fb9587048bb0296de
 │   ├── info
 │   └── pack
 └── refs
```
如您所见，这会导致两个主要变化。它修改的第一件事是`index`文件。 [索引](https://git-scm.com/docs/index-format)用于存储有关当前暂存的内容的信息。这用于表示名为 `file` 的文件已添加到索引中。

第二个也是更重要的变化是添加了一个新的文件夹 `objects/4c` 和 `5b58f323d7b459664b5d3fb9587048bb0296de` 其中的文件。

### 但是那个文件里有什么？
在这里，我们详细介绍`了 git` 如何存储事物。让我们从看看其中存在什么样的数据开始。

```
$ file .git/objects/4c/5b58f323d7b459664b5d3fb9587048bb0296de
.git/objects/4c/5b58f323d7b459664b5d3fb9587048bb0296de: zlib compressed data
```

嗯，但是什么是 zlib 压缩数据？
```
$ zlib-flate -uncompress <.git/objects/4c/5b58f323d7b459664b5d3fb9587048bb0296de
blob 9\0meain.io
```
看起来它包含我们做了一个 `git add`的名为 `file` 的文件的类型、大小和数据。在这种情况下，数据显示它是一个大小为 `9` 的 `blob`，内容是 `meain.io`。

### 好的，但是那个文件名是什么

嗯，好问题。它来自内容的 sha1。如果您获取 zlib 压缩数据并通过 `sha1sum` 进行管道传输，您将获得文件名。
```
$ zlib-flate -uncompress <.git/objects/4c/5b58f323d7b459664b5d3fb9587048bb0296de|sha1sum
4c5b58f323d7b459664b5d3fb9587048bb0296de
```

`git` 获取要写入的内容的 sha1，获取前两个字符，在本例中为 `4c`，创建一个文件夹，然后使用其余部分作为文件名。`Git` 从前两个字符创建文件夹，以确保我们在单个`对象`文件夹下没有太多文件。

### 向 `git cat-file`[#](https://blog.meain.io/2023/what-is-in-dot-git/#say-hello-to-git-cat-file) 问好
事实上，由于这是 git 中更重要的部分之一，git 也有一个管道命令来查看对象的内容。您可以使用 `git cat-file` 将 `-t` 用于类型，`-s` 用于大小，将 `-p` 用于内容。

```
$ git cat-file -t 4c5b58f323d7b459664b5d3fb9587048bb0296de
blob

$ git cat-file -s 4c5b58f323d7b459664b5d3fb9587048bb0296de
9

$ git cat-file -p 4c5b58f323d7b459664b5d3fb9587048bb0296de
meain.io
```

## 让我们提交

现在我们知道添加文件时会发生什么变化，让我们通过提交将其提升到一个新的水平。

```bash
$ git commit -m 'Initial commit'[master (root-commit) 4c201df] Initial commit 1 file changed, 1 insertion(+) create mode 100644 file
```
以下是更改内容：
```
--- init        2024-07-02 15:14:00.584674816 +0530
+++ commit      2023-07-02 15:33:28.536144046 +0530
@@ -1,11 +1,25 @@
 .git
+├── COMMIT_EDITMSG
 ├── config
 ├── HEAD
 ├── hooks
 │   └── prepare-commit-msg.msample
 ├── index
+├── logs
+│   ├── HEAD
+│   └── refs
+│       └── heads
+│           └── master
 ├── objects
+│   ├── 3c
+│   │   └── 201df6a1c4d4c87177e30e93be1df8bfe2fe19
 │   ├── 4c
 │   │   └── 5b58f323d7b459664b5d3fb9587048bb0296de
+│   ├── 62
+│   │   └── 901ec0eca9faceb8fe0a9870b9b6cde75a9545
 │   ├── info
 │   └── pack
 └── refs
     ├── heads
+    │   └── master
     └── tags
```

哇，看起来有很多变化。让我们一一介绍一下。第一个是新文件 `COMMIT_EDITMSG`。顾名思义，它包含（最后的）提交消息。

_如果您在没有 `-m` 标志的情况下运行 `git commit` 命令，则 `git` 获取提交消息的方式是打开一个带有 `COMMIT_EDITMSG` 文件的编辑器，让用户编辑提交消息，一旦用户更新了它并退出编辑器，`git` 使用文件的内容作为提交消息。_

它还添加了一个全新的文件夹`log` 。这是 git 在存储库中记录所有提交更改的一种方式。您将能够在此处看到所有引用和 `HEAD` 的提交更改。

`objects`目录也进行了一些更改，但我希望您首先查看我们现在拥有文件``master``的 `refs/heads` 目录。正如您可能已经猜到的那样，这是对``master``分支的引用。让我们看看里面有什么。


```bash
$ cat refs/heads/master 3c201df6a1c4d4c87177e30e93be1df8bfe2fe19
```

看起来它指向其中一个新对象。我们知道如何看物体，让我们这样做。

```shell
$ git cat-file -t 3c201df6a1c4d4c87177e30e93be1df8bfe2fe19
commit

$ git cat-file -p 3c201df6a1c4d4c87177e30e93be1df8bfe2fe19
tree 62902ec0eca9faceb8fe0a9870b9b6cde75a9545
author Abin Simon <mail@meain.io> 1688292123 +0530
committer Abin Simon <mail@meain.io> 1688292123 +0530

Initial commit
```
嗯，看起来那是一种新的物体。这似乎是一个`commit`对象。 `commit`对象的内容告诉我们它包含一个带有哈希的`tree`对象 `62902ec0eca9faceb8fe0a9870b9b6cde75a9545` ，它看起来像我们执行提交时添加的另一个对象。 `commit`对象还包含有关作者和提交者是谁的信息，在本例中都是我。最后还显示了此提交的提交消息是什么。


现在让我们看看`tree`对象包含什么。
```
$ git cat-file -t 62902ec0eca9faceb8fe0a9870b9b6cde75a9545
tree

$ git cat-file -p 62901ec0eca9faceb8fe0a9870b9b6cde75a9545
100644 blob 4c5b58f323d7b459664b5d3fb9587048bb0296de    file
```

`tree`对象将以其他树和 blob 对象的形式包含工作目录的状态。在这种情况下，由于我们只有一个名为 `file` 的文件，因此您只会看到一个对象。如果您看到，该文件指向我们在执行 `git add file`时添加的原始对象。

这是更成熟的存储库的树的样子。从`提交`对象链接的`树`对象内使用了更多的`树`对象来表示文件夹。

```
$ git cat-file -p 2e5e84c3ee1f7e4cb3f709ff5ca0ddfc259a8d04
100644 blob 3cf56579491f151d82b384c211cf1971c300fbf8    .dockerignore
100644 blob 02c348c202dd41f90e66cfeb36ebbd928677cff6    .gitattributes
040000 tree ab2ba080c4c3e4f2bc643ae29d5040f85aca2551    .github
100644 blob bdda0724b18c16e69b800e5e887ed2a8a210c936    .gitignore
100644 blob 3a592bc0200af2fd5e3e9d2790038845f3a5cf9b    CHANGELOG.md
100644 blob 71a7a8c5aacbcaccf56740ce16a6c5544783d095    CODE_OF_CONDUCT.md
100644 blob f433b1a53f5b830a205fd2df78e2b34974656c7b    LICENSE
100644 blob 413072d502db332006536e1af3fad0dce570e727    README.md
100644 blob 1dd7ed99019efd6d872d5f6764115a86b5121ae9    SECURITY.md
040000 tree 918756f1a4e5d648ae273801359c440c951555f9    build
040000 tree 219a6e58af53f2e53b14b710a2dd8cbe9fea15f5    design
040000 tree 5810c119dd4d9a1c033c38c12fae781aeffeafc1    docker
040000 tree f09c5708676cdca6562f10e1f36c9cfd7ee45e07    src
040000 tree e6e1595f412599d0627a9e634007fcb2e32b62e5    website

```
# 进行更改 [#](https://blog.meain.io/2023/what-is-in-dot-git/#making-a-change)

让我们对文件进行更改，看看它是如何工作的。
```
$ echo 'blog.meain.io' > file
$ git commit -am 'Use blog link'
[master 68ed5aa] Use blog link
 1 file changed, 1 insertion(+), 1 deletion(-)
```

这是它的作用：
```
--- commit      2024-07-02 15:33:28.536144046 +0530
+++ update      2023-07-02 15:47:20.841154907 +0530
@@ -17,6 +17,12 @@
 │   │   └── 5b58f323d7b459664b5d3fb9587048bb0296de
 │   ├── 62
 │   │   └── 901ec0eca9faceb8fe0a9870b9b6cde75a9545
+│   ├── 67
+│   │   └── ed5aa2372445cf2249d85573ade1c0cbb312b1
+│   ├── 8a
+│   │   └── b377e2f9acd9eaca12e750a7d3cb345065049e
+│   ├── e5
+│   │   └── ec63cd761e6ab9d11e7dc2c4c2752d682b36e2
 │   ├── info
 │   └── pack
 └── refs
```
好吧，我们添加了 3 个新对象。其中一个是具有文件新内容的 `blob` 对象，一个是`树`对象，最后一个是`提交`对象。

让我们从 `HEAD` 或 `refs/heads/master` 再次跟踪它们。
```shell
$ git cat-file -p refs/heads/master
tree 9ab377e2f9acd9eaca12e750a7d3cb345065049e
parent 3c201df6a1c4d4c87177e30e93be1df8bfe2fe19
author Abin Simon <mail@meain.io> 1688292975 +0530
committer Abin Simon <mail@meain.io> 1688292975 +0530

Use blog link

$ git cat-file -p 9ab377e2f9acd9eaca12e750a7d3cb345065049e
100644 blob e5ec63cd761e6ab9d11e7dc2c4c2752d682b36e2    file

$ git cat-file -p e6ec63cd761e6ab9d11e7dc2c4c2752d682b36e2
blog.meain.io
```

那些注意的人可能已经注意到， `提交`对象现在有一个名为 `parent` 的附加键，该键链接到上一个提交，因为此提交是在上一个提交之上创建的。

# 创建分支 [#](https://blog.meain.io/2023/what-is-in-dot-git/#creating-a-branch)

大约是时候我们创建了一个分支了。让我们用 `git branch fix-url` 来做到这一点。

```
--- update      2024-07-02 15:47:20.841154907 +0530
+++ branch      2023-07-02 15:55:25.165204941 +0530
@@ -27,5 +28,6 @@
 │   └── pack
 └── refs
     ├── heads
+    │   ├── fix-url
     │   └── master
     └── tags
```
这会在文件夹 `refs/heads` 下添加一个新文件，其中文件作为分支名称，内容作为最新提交的 ID。

```
$ cat .git/refs/heads/fix-url
68ed5aa2372445cf2249d85573ade1c0cbb312b1
```

这几乎就是创建分支的全部内容。`git` 中的分支真的很便宜。标签的行为方式也相同，只是它们是在 `refs/tags` 下创建的。

在 `logs` 目录下还添加了一个文件，用于存储类似于 `master` 分支的提交历史数据。