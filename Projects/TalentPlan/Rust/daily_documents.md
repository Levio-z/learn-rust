## Day 1 2025/05/25

### 计划&进展
- rust
	- Building Blocks 2
		- [x] http://blog.notdot.net/2009/12/Damn-Cool-Algorithms-Log-structured-storage
			- [x] 操作系统导论第43章 日志结构文件系统
		- [ ] The Design and Implementation of a Log-Structured File System
			- [x] 使用软件翻译，明天阅读打下基础
### 收获
- 了解了日志结构文件系统的设计，核心就是提升写的性能，将随机写改为顺序写，读的话因为有缓存所以性能瓶颈都在写上，并且日志结构文件系统的恢复很方便，只要从上次保存的最新有效的写入节点按照写入的时间往后读取就可以了。数据是日志也是底层数据结构。

### 问题
- 整个论文比较复杂的，明天在梳理一遍，整个实现是弄懂了，但是没太接触传统的文件系统。不知道传统文件系统的设计。需要补充这方面的知识。