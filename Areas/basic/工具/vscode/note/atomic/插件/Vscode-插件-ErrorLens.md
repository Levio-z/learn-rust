## 1. 核心观点  
### Ⅰ. 概念层

##  简介

**ErrorLens** 通过让诊断信息更醒目来强化语言诊断功能。  
当语言服务器产生诊断信息（如错误、警告）时，ErrorLens 会高亮整行，并在行尾显示错误信息。

## 功能概览

- 高亮包含诊断的整行
- 在行尾追加诊断信息
- 在编辑器侧边栏（gutter）显示图标
- 在状态栏显示错误信息

### Ⅱ. 应用层

#### 在上面显示，提供修复按钮
![image.png](https://raw.githubusercontent.com/Levio-z/MyPicture/main/img/20260120122601364.png)

![image.png](https://raw.githubusercontent.com/Levio-z/MyPicture/main/img/20260120122613976.png)

### 列出的诊断等级
- 全选好了
![image.png](https://raw.githubusercontent.com/Levio-z/MyPicture/main/img/20260120123739607.png)
### Source code
- 快速复制信息
![image.png](https://raw.githubusercontent.com/Levio-z/MyPicture/main/img/20260120125412608.png)
### 显示状态栏图标
- 建议打开
![image.png](https://raw.githubusercontent.com/Levio-z/MyPicture/main/img/20260120130306220.png)
![image.png](https://raw.githubusercontent.com/Levio-z/MyPicture/main/img/20260120130321207.png)
只显示当前编辑器的内容

![image.png](https://raw.githubusercontent.com/Levio-z/MyPicture/main/img/20260120131619953.png)
### 状态栏显示错误
- 建议开启，查错误挺好用的
![image.png](https://raw.githubusercontent.com/Levio-z/MyPicture/main/img/20260120132535300.png)

就不显示具体信息了，配置这三个
![image.png](https://raw.githubusercontent.com/Levio-z/MyPicture/main/img/20260120133446514.png)

默认显示最近，最不安全的
![image.png](https://raw.githubusercontent.com/Levio-z/MyPicture/main/img/20260120133510762.png)

### Ⅲ. 实现层

### **IV**.原理层


## 2. 背景/出处  
- 来源：
- 引文/摘要：  
  - …  
  - …  

## 3. 展开说明  

### Error Lens: Editor Hover Parts Enabled
- **`messageEnabled`**
    
    控制悬浮提示框中是否显示**问题的详细描述信息**。
    
    - 开启后：你会看到错误 / 警告的具体原因，比如 “变量未定义” 或 “语法错误”。
    - 关闭后：悬浮框里就不会显示这段文字，只能看到代码本身。
    
- **`sourceCodeEnabled`**
    
    控制悬浮提示框中是否显示**出现问题的源代码片段**。
    
    - 开启后：会在提示框里把出错的代码行也展示出来，方便你直接对照。
    - 关闭后：提示框里只会有问题描述，不会附带代码。
    
- **`buttonsEnabled`**
    
    控制悬浮提示框底部是否显示**快捷操作按钮**。
    
    - 开启后：会出现如 “快速修复”、“查看问题详情” 等按钮，让你一键处理问题。
    - 关闭后：提示框就只是纯文本信息，没有可点击的操作按钮。

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
