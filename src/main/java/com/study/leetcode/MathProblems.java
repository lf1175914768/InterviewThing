package com.study.leetcode;

import java.util.PriorityQueue;

/**
 * <p>description: 数学技巧类的题目  </p>
 * <p>className:  MathProblems </p>
 * <p>create time:  2022/5/13 13:47 </p>
 *
 * @author feng.liu
 * @since 1.0
 **/
public class MathProblems {

    // -------组合总和 start >>--------

    /**
     * 给你一个整数，请你找出并返回第 n 个丑数。
     * 丑数就是只包含质因数 2、3和/或 5 的正整数。
     *
     * 对应 leetcode 中第 264 题。
     */
    public int nthUglyNumber(int n) {
        PriorityQueue<Integer> queue = new PriorityQueue<>();
        int res = 1;
        for (int i = 1; i < n; i++) {
            queue.offer(res * 2);
            queue.offer(res * 3);
            queue.offer(res * 5);
            res = queue.poll();
            while (!queue.isEmpty() && queue.peek() == res) {
                queue.poll();
            }
        }
        return res;
    }

    // -------组合总和 << end --------
}
