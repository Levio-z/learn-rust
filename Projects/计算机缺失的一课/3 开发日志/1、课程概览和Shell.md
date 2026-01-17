---
tags:
  - fleeting
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
Shell是你与计算机交互的主要方式之一，一旦你想用电脑做更多的事情，而不仅仅是使用现有的可视化界面，那么你就需要更强大的电脑了。你可能习惯使用可视化界面，但是这些界面在功能上存在一些局限性。还有其他不同的组合方式和编程方式实现它们的自动化。命令行或基于文本的工具，以及shell就是你执行操作的地方。

`\` 会让空格失去分隔意义

空格分隔参数

电脑自带内置程序

包含大量以终端为中心的应用程序，你的文件系统和shell都有办法确定文件的位置

invariant environment variable

每次启动shell设置，你的主目录在哪里，你的用户名
$Path 就是路径变量，计算机上搜索所有可执行文件的地方
输入可执行文件，计算机会搜索这些目录查找可匹配的文件或程序

which echo

/是分隔符

绝对路径
完全确定文件位置的路径

相对路径是想对你你当前路径的

pwd

cd 更改当前目录位置

ls 列出当前目录下的文件

ls命令打开那个文件而不是当前目录

~会带你到主目录

-会带你返回上一次目录

-选项和 参数 --

文件的权限

读取是能否列出列表
对目录拥有写入权限是指你是否有权重命名、创建或删除文件。

对文件有写，但是目录没有写，你可以清空文件，但是不能删除文件

**目录的“执行权限（x）”并不是“能不能运行文件”，而是：**

> **是否允许“进入该目录、解析路径、访问其内部 inode”**

没有目录的 `x` 权限，**目录里的任何东西对你来说都“不可达”**。

rmdir 只删除空目录

ctrl + l 清空

每个程序都有两个流

```
missing:~$ echo hello > hello.txt
missing:~$ cat hello.txt
hello
missing:~$ cat < hello.txt
hello
missing:~$ cat < hello.txt > hello2.txt
missing:~$ cat hello2.txt
hello
```
```
>>追加而不是覆盖
>>
>>cat < hello.txt >> hello2.txt
```

将左侧的输出作为右侧的输入

```
missing:~$ ls -l / | tail -n1
drwxr-xr-x 1 root  root  4096 Jun 20  2019 var
missing:~$ curl --head --silent google.com | grep --ignore-case content-length | cut --delimiter=' ' -f2
219
```


```
missing:~$ ls -l / | tail -n1
drwxr-xr-x 1 root  root  4096 Jun 20  2019 var
missing:~$ curl --head --silent google.com | grep --ignore-case content-length | cut --delimiter=' ' -f2
219
```

root用户
对设备执行任何操作

sudo su

tee接收输入并写入文件

xdg-open
## 4. 与其他卡片的关联  
- 前置卡片：
- 后续卡片：
- 相似主题：

## 5. 应用/启发  
- 可以如何应用在工作、学习、生活中  
- 引发的思考与问题  

## 6. 待办/进一步探索  
 
  
