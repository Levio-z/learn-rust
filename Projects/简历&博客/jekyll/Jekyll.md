[Jekyll](https://jekyllrb.com/) 是一个在 Ruby 编程语言上运行的静态站点生成器。您可能听说过 Jekyll 或静态站点生成器，但不知道如何或从哪里开始。本指南旨在成为一个完整的教程，不需要额外的资源来帮助您启动和运行 Jekyll。

虽然 Jekyll 被宣传为博客平台，但它也可以用于静态网站，就像 WordPress 一样。Jekyll 利用 [Markdown](https://daringfireball.net/projects/markdown/) 的力量，使编写 HTML 变得更加容易和高效。此外，Jekyll 内置了 Sass，如果您从未使用过 CSS 预处理器，那么现在是学习的好时机。如果您已经知道如何使用 Sass，您会有宾至如归的感觉。

>**Sass（Syntactically Awesome Stylesheets）**：一种 CSS 预处理语言，提供变量、嵌套、Mixin、继承等特性，提升 CSS 的组织性与复用性。
>**CSS 预处理器**：在标准 CSS 编译之前的扩展语法，允许开发者以更结构化的方式编写样式。

This is what the website we make will look like:  
这就是我们制作的网站的样子：
![](asserts/Pasted%20image%2020250731175000.png)

### Prerequisites  先决条件

- Basic knowledge of HTML and CSS  
    HTML 和 CSS 的基本知识
- [Basic command line knowledge  
    基本的命令行知识](https://www.taniarascia.com/how-to-use-the-command-line-for-apple-macos-and-linux/)
- A GitHub account  GitHub 帐户


### 什么是静态站点生成器？

静态站点生成器使用纯 HTML 文件构建网站。当用户访问由静态站点生成器创建的网站时，它的加载方式与使用纯 HTML 创建网站没有什么不同。相比之下，每次用户访问站点时，都必须构建在服务器端语言（如 PHP）上运行的动态站点.

您可以将静态站点生成器视为一种非常简单的 CMS（内容管理系统）。例如，不必在每个页面上包含整个页眉和页脚，您可以创建一个 header.html 和 footer.html 并将它们加载到每个页面中。**您不必用 HTML 编写，而是可以用 Markdown 编写，这更快、更高效。**

>例如，您无需在每个页面中手动重复书写页眉（header）和页脚（footer）HTML，只需将它们分别定义为 `header.html` 和 `footer.html` 模板文件，并在各页面中通过 `include` 指令进行复用。  
此外，与传统 HTML 编写方式不同，静态站点生成器支持使用 Markdown 编写内容。这种方式更简洁、更高效，尤其适合撰写博客、文档或技术笔记。

### 优势


 - **Speed** - your website will perform much faster, as the server does not need to parse any content. It only needs to read plain HTML.  
    **速度** - 您的网站将运行得更快，因为服务器不需要解析任何内容。它只需要读取纯 HTML。
- **Security** - your website will be much less vulnerable to attacks, since there is nothing that can be exploited server side.  
    **安全性** - 您的网站将不易受到攻击，因为服务器端没有任何可以利用的东西。
- **Simplicity** - there are no databases or programming languages to deal with. A simple knowledge of HTML and CSS is enough.  
    **简单** - 无需处理数据库或编程语言。简单的 HTML 和 CSS 知识就足够了。
- **Flexibility** - you know exactly how your site works, as you made it from scratch.  
    **灵活性** - 您确切地知道您的网站是如何工作的，因为您从头开始制作它。

当然，动态网站也有其优势。添加管理面板便于更新，特别是对于那些不懂技术的人来说。通常，静态站点生成器不是为客户制作 CMS 的最佳主意。静态站点生成器也无法使用实时内容进行更新。重要的是要了解两者的工作原理，以了解什么最适合您的特定项目。

###  Installing Jekyll  安装 Jekyll
[2 docker安装Jekll](ChatGpt/2%20docker安装Jekll.md)


### 创建 Jekyll 主题

使用 Jekyll，我们将能够将 SCSS （Sass） 文件处理为 CSS （**.scss** -> **.css**），将 Markdown 处理为 HTML （**.md** -> **.html**）。不需要额外的任务运行器或终端命令！

关于 Jekyll 文件系统，有一些重要的事情需要了解。

- “分发”文件夹称为 **_site**。这就是**静态站点生成器生成的内容！** **切勿**在该文件夹中放置任何文件;它们将被删除和覆盖。
- **_sass** 文件夹用于 Sass 分部。这里的每个文件都应该以下划线开头，它将编译到 **css** 文件夹中。
- 放置在主目录中的任何文件或文件夹都将按原样编译到 **_site** 目录中。

### 配置

在主目录中，有一个名为 **_config.yml** 的文件。它看起来像这样：

```yaml
# Site settings
title: Your awesome title
email: your-email@domain.com
description: > # this means to ignore newlines until "baseurl:"
  Write an awesome description for your new site here. You can edit this
  line in _config.yml. It will appear in your document head meta (for
  Google search results) and in your feed.xml site description.
baseurl: '' # the subpath of your site, e.g. /blog/
url: 'http://yourdomain.com' # the base hostname & protocol for your site
twitter_username: jekyllrb
github_username: jekyll

# Build settings
markdown: kramdown
```
- 对 **_config.yml** 所做的更改不会被 `jekyll serve` 监视。您必须在更改任何配置后重新启动并保留 Jekyll。
- 所有缩进都是强制性的，必须使用两个空格，否则文件将无法工作。
我将对配置进行一些更改。
```yaml
# Site Settings
title: Start Jekyll
email: email@gmail.com
description: >
  A guide to getting started with Jekyll.
baseurl: ''
url: 'http://localhost:4000'
twitter_username: taniarascia
github_username: taniarascia
# Build Settings
sass:
  sass_dir: _sass
include: ['_pages']
kramdown:
  input: GFM
```