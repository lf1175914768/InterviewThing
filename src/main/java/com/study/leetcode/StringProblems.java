package com.study.leetcode;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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

    // -------判断子序列 start >>--------

    /**
     * 给定字符串 s 和 t ，判断 s 是否为 t 的子序列。
     * 字符串的一个子序列是原始字符串删除一些（也可以不删除）字符而不改变剩余字符相对位置形成的新字符串。（例如，"ace"是"abcde"的一个子序列，而"aec"不是）。
     *
     * 对应 leetcode 中第 392 题。
     */
    public boolean isSubsequence(String s, String t) {
        int cur = 0, len = s.length();
        for (int i = 0; i < t.length(); i++) {
            if (len == cur) return true;
            if (t.charAt(i) == s.charAt(cur)) {
                cur++;
            }
        }
        return cur == len;
    }

    /**
     * 以空间换取时间，预先处理 字符串 t，当有很多的 s 判断是否是 t 的子序列时，效率比较高
     */
    public boolean isSubsequence_v2(String s, String t) {
        int tLen = t.length(), sLen = s.length();
        Map<Character, List<Integer>> cache = new HashMap<>();
        for (int i = 0; i < tLen; i++) {
            char ch = t.charAt(i);
            List<Integer> list = cache.computeIfAbsent(ch, k -> new ArrayList<>());
            list.add(i);
        }
        int target = 0;
        for (int i = 0; i < sLen; i++) {
            char ch = s.charAt(i);
            List<Integer> list = cache.get(ch);
            if (list == null) return false;
            int left = 0, right = list.size();
            while (left < right) {
                int mid = left + ((right - left) >>> 1);
                if (list.get(mid) < target) {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }
            if (left == list.size()) return false;
            target = list.get(left) + 1;
        }
        return true;
    }

    // -------判断子序列 << end --------
}
