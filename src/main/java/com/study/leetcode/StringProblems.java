package com.study.leetcode;

import java.util.List;

/**
 * String problems
 */
public class StringProblems {

    // -------使括号有效的最少添加 start >>--------

    /**
     * 只有满足下面几点之一，括号字符串才是有效的：
     *
     * 它是一个空字符串，或者
     * 它可以被写成 AB （A 与 B 连接）, 其中 A 和 B 都是有效字符串，或者
     * 它可以被写作 (A)，其中 A 是有效字符串。
     * 给定一个括号字符串 s ，移动N次，你就可以在字符串的任何位置插入一个括号。
     *
     * 例如，如果 s = "()))" ，你可以插入一个开始括号为 "(()))" 或结束括号为 "())))" 。
     * 返回 为使结果字符串 s 有效而必须添加的最少括号数。
     *
     * 对应 leetcode 中第 921 题。
     */
    public int minAddToMakeValid(String s) {
        int need = 0, res = 0;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '(') {
                // 对右括号的需求 + 1
                need++;
            } else if (c == ')') {
                need--;

                if (need == -1) {
                    // 需要插入一个左括号
                    res++;
                    // 重新将 need 设置为 0
                    need++;
                }
            }
        }
        return need + res;
    }

    // -------使括号有效的最少添加 << end --------
}
