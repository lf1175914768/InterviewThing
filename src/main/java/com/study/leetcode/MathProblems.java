package com.study.leetcode;

import java.util.*;

/**
 * <p>description: 数学技巧类的题目  </p>
 * <p>className:  MathProblems </p>
 * <p>create time:  2022/5/13 13:47 </p>
 *
 * @author feng.liu
 * @since 1.0
 **/
public class MathProblems {

    // -------丑数II start >>--------

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

    // -------丑数II << end --------

    // -------整数转换 start >>--------

    /**
     * 给定一个正整数 n ，你可以做如下操作：
     * 如果 n 是偶数，则用 n / 2替换 n 。
     * 如果 n 是奇数，则可以用 n + 1或n - 1替换 n 。
     * 返回 n 变为 1 所需的 最小替换次数 。
     *
     * 使用 dfs 的方法进行解答。
     *
     * 对应 leetcode 中第 397 题。
     */
    public int integerReplacement(int n) {
        Map<Integer, Integer> cache = new HashMap<>();
        return integerReplacementDfs(n, cache);
    }

    private int integerReplacementDfs(int n, Map<Integer, Integer> cache) {
        if (n == 1) return 0;
        if (cache.containsKey(n)) return cache.get(n);
        int ans;
        if ((n & 1) == 0) {
            ans = integerReplacementDfs(n / 2, cache) + 1;
        } else {
            ans = Math.min(integerReplacementDfs(n + 1, cache), integerReplacementDfs(n - 1, cache)) + 1;
        }
        cache.put(n, ans);
        return ans;
    }

    // -------整数转换 << end --------

    // -------字典序排数 start >>--------

    /**
     * 给你一个整数 n ，按字典序返回范围 [1, n] 内所有整数。
     * 你必须设计一个时间复杂度为 O(n) 且使用 O(1) 额外空间的算法。
     *
     * 使用 递归的方法进行解答。
     *
     * 对应 leetcode 中第 386 题。
     */
    public List<Integer> lexicalOrder(int n) {
        List<Integer> res = new ArrayList<>();
        for (int i = 1; i <= 9; i++) {
            lexicalOrderDfs(i, n, res);
        }
        return res;
    }

    private void lexicalOrderDfs(int cur, int limit, List<Integer> res) {
        if (cur > limit) return;
        res.add(cur);
        for (int i = 0; i <= 9; i++) {
            int next = cur * 10 + i;
            if (next <= limit) {
                lexicalOrderDfs(next, limit, res);
            }
        }
    }

    /**
     * 使用迭代的方式进行解答。
     */
    public List<Integer> lexicalOrder_v2(int n) {
        List<Integer> res = new ArrayList<>();
        for (int i = 0, j = 1; i < n; i++) {
            res.add(j);
            if (j * 10 <= n) {
                j *= 10;
            } else {
                while (j % 10 == 9 || j == n) {
                    j /= 10;
                }
                j++;
            }
        }
        return res;
    }

    // -------字典序排数 << end --------
}
