# 二分法
## 总结
1 本题使用什么方法

根据本题题目选定模板

### 2 二分法解题模板

- 经典二分法（左闭右开）
    - [704.二分查找](https://www.notion.so/704-1d755e4de8248030a00aecbda2bacb70?pvs=21)
    - 返回right就是插入位置
    - left-1一定是比目标值要小的，right一定是比目标值大的
    - left就一定是比目标值小的元素集合的下一个元素
    - left和right相交就表明，小的元素，和大的元素边界衔接了，此时left就是大于目标值元素的第一个元素位置，也是所有小于目标值元素的下一个位置

### 3 需要考虑的边界

- 可能的减枝：
    - // 1.考虑边界：数组为空、全小于、全大于
    - // 2.if nums[left] != target {
    - 相关题目
        - [**34. 在排序数组中查找元素的第一个和最后一个位置**](https://www.notion.so/34-30226477874c465ba59e7ddda63fec9b?pvs=21)

### 4 考虑溢出

- [[69. x 的平方根](https://leetcode.cn/problems/sqrtx/)]([https://www.notion.so/69-x-a4b0932ddde440cebaf49a51e99abe2e?pvs=21](https://www.notion.so/69-x-a4b0932ddde440cebaf49a51e99abe2e?pvs=21))
- [[**367. 有效的完全平方数**](https://leetcode.cn/problems/valid-perfect-square/)]([https://www.notion.so/367-8541c5308f2448d884a8e8316d421b8d?pvs=21](https://www.notion.so/367-8541c5308f2448d884a8e8316d421b8d?pvs=21))
## 题目
- 