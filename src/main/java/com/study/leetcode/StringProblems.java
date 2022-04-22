package com.study.leetcode;

import java.util.*;

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

    // -------比较版本号 start >>--------

    /**
     * 给你两个版本号 version1 和 version2 ，请你比较它们。
     * 版本号由一个或多个修订号组成，各修订号由一个 '.' 连接。每个修订号由 多位数字 组成，可能包含 前导零 。每个版本号至少包含一个字符。
     * 修订号从左到右编号，下标从 0 开始，最左边的修订号下标为 0 ，下一个修订号下标为 1 ，以此类推。例如，2.5.33 和 0.1 都是有效的版本号。
     * 比较版本号时，请按从左到右的顺序依次比较它们的修订号。比较修订号时，只需比较 忽略任何前导零后的整数值 。也就是说，修订号 1 和修订号 001 相等 。
     * 如果版本号没有指定某个下标处的修订号，则该修订号视为 0 。例如，版本 1.0 小于版本 1.1 ，因为它们下标为 0 的修订号相同，而下标为 1 的修订号分别为 0 和 1 ，0 < 1 。
     * 返回规则如下：
     *
     * 如果 version1 > version2 返回 1，
     * 如果 version1 < version2 返回 -1，
     * 除此之外返回 0。
     *
     * 对应 leetcode 中第 165 题。
     */
    public int compareVersion(String version1, String version2) {
        String[] arr1 = version1.split("\\.");
        String[] arr2 = version2.split("\\.");
        int minLength = Math.min(arr1.length, arr2.length);
        for (int i = 0; i < minLength; i++) {
            int rs = compareVersionCompare(arr1[i], arr2[i]);
            if (rs != 0) {
                return rs;
            }
        }
        for (int i = minLength; i < arr1.length; i++) {
            if (compareVersionNotAllZero(arr1[i])) return 1;
        }
        for (int i = minLength; i < arr2.length; i++) {
            if (compareVersionNotAllZero(arr2[i])) return -1 ;
        }
        return 0;
    }

    private boolean compareVersionNotAllZero(String str) {
        for (int i = 0; i < str.length(); i++) {
            if (str.charAt(i) != '0') return true;
        }
        return false;
    }

    private int compareVersionCompare(String str1, String str2) {
        int num1 = 0, num2 = 0;
        for (int i = 0; i < str1.length(); i++) {
            if (str1.charAt(i) != '0') {
                num1 = Integer.parseInt(str1.substring(i));
                break;
            }
        }
        for (int i = 0; i < str2.length(); i++) {
            if (str2.charAt(i) != '0') {
                num2 = Integer.parseInt(str2.substring(i));
                break;
            }
        }
        return Integer.compare(num1, num2);
    }

    // -------比较版本号 << end --------

    // -------不同的子序列 start >>--------

    /**
     * 给定一个字符串 s 和一个字符串 t ，计算在 s 的子序列中 t 出现的个数。
     * 字符串的一个 子序列 是指，通过删除一些（也可以不删除）字符且不干扰剩余字符相对位置所组成的新字符串。（例如，"ACE" 是 "ABCDE" 的一个子序列，而 "AEC" 不是）
     * 题目数据保证答案符合 32 位带符号整数范围。
     *
     * 思路：动态规划
     * dp[i][j] 代表 t 中前 i 字符串可以由 s 中前 j 个字符串组成组多个数
     * 所以有两种情况：
     * case 1：s[j] == t[i]时，dp[i][j] 由两部分组成：
     *           如果 s[j] 和 t[i] 匹配，则考虑 t[i - 1] 作为 s[j - 1]的子序列，子序列数为 dp[i - 1][j - 1];
     *           如果 s[j] 不和 t[i] 匹配，则考虑 t[i] 作为 s[j - 1] 的子序列，子序列数为 dp[i][j - 1];
     * case 2: s[j] != t[i] 时，s[j] 不能和 t[i] 匹配，因此只考虑 t[i] 作为 s[j - 1] 的子序列，子序列数为 dp[i][j - 1]。
     *
     * 因此可以得到如下状态转移方程：
     *            | dp[i][j] = dp[i - 1][j - 1] + dp[i][j - 1],      t[i] == s[j],
     * dp[i][j] = |
     *            | dp[i][j] = dp[i][j - 1],                         t[i] != s[j],
     *
     * 考虑边界情况：
     * 当 i == 0时，t[i]为空字符串，由于空字符串是任何字符的子序列，因此对任意 0 <= j <= m, 有 dp[i][m] = 1；
     * 当 j == 0时，s[j]为空字符串，由于空字符串不是任何非空字符串的子序列，因此对任意 0 < i <= n, 有 dp[n][j] = 0;
     *
     * 对应 leetcode 中第 115 题。
     */
    public int numDistinct(String s, String t) {
        int[][] dp = new int[t.length() + 1][s.length() + 1];
        for (int j = 0; j <= s.length(); j++)
            dp[0][j] = 1;
        for (int i = 1; i <= t.length(); i++) {
            for (int j = 1; j <= s.length(); j++) {
                if (t.charAt(i - 1) == s.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + dp[i][j - 1];
                } else {
                    dp[i][j] = dp[i][j - 1];
                }
            }
        }
        return dp[t.length()][s.length()];
    }

    // -------不同的子序列 << end --------

    // -------颠倒字符串中的单词 start >>--------

    /**
     * 给你一个字符串 s ，颠倒字符串中 单词 的顺序。
     * 单词 是由非空格字符组成的字符串。s 中使用至少一个空格将字符串中的 单词 分隔开。
     * 返回 单词 顺序颠倒且 单词 之间用单个空格连接的结果字符串。
     * 注意：输入字符串 s中可能会存在前导空格、尾随空格或者单词间的多个空格。返回的结果字符串中，单词间应当仅用单个空格分隔，且不包含任何额外的空格。
     *
     * 对应 leetcode 中第 151 题。
     */
    public String reverseWords(String s) {
        char[] arr = s.toCharArray();
        int left = 0, right = arr.length - 1;
        while (left < right && arr[left] == ' ') left++;
        while (left < right && arr[right] == ' ') right--;
        StringBuilder sb = new StringBuilder();
        while (left <= right) {
            int index = right;
            while (index >= left && arr[index] != ' ') index--;
            for (int i = index + 1; i <= right; i++) {
                sb.append(arr[i]);
            }
            if (index > left) sb.append(' ');
            while (index >= left && arr[index] == ' ') index--;
            right = index;
        }
        return sb.toString();
    }

    // -------颠倒字符串中的单词 << end --------

    // -------复原IP地址 start >>--------

    /**
     * 有效 IP 地址 正好由四个整数（每个整数位于 0 到 255 之间组成，且不能含有前导 0），整数之间用 '.' 分隔。
     * 例如："0.1.2.201" 和 "192.168.1.1" 是 有效 IP 地址，但是 "0.011.255.245"、"192.168.1.312" 和 "192.168@1.1" 是 无效 IP 地址。
     * 给定一个只包含数字的字符串 s ，用以表示一个 IP 地址，返回所有可能的有效 IP 地址，这些地址可以通过在 s 中插入 '.' 来形成。
     * 你 不能 重新排序或删除 s 中的任何数字。你可以按 任何 顺序返回答案。
     *
     * 使用 回溯的 算法思想进行解答。
     *
     * 对应 leetcode 中第 93 题。
     */
    public List<String> restoreIpAddresses(String s) {
        int len = s.length();
        List<String> res = new ArrayList<>();
        if (len > 12 || len < 4) return res;
        Deque<String> path = new ArrayDeque<>();
        restoreIpAddressesDfs(s, 0, 4, path, res);
        return res;
    }

    private void restoreIpAddressesDfs(String s, int begin, int residue, Deque<String> path, List<String> res) {
        if (begin == s.length()) {
            if (residue == 0) {
                res.add(String.join(".", path));
            }
            return;
        }
        for (int i = begin; i < begin + 3; i++) {
            if (i >= s.length()) break;
            // 每段最多只能截取3个数，如果字符串剩余长度大于分段所需最大长度，直接continue
            if (residue * 3 < s.length() - i) {
                continue;
            }
            if (restoreIpAddressesJudgeIpSegment(s, begin, i)) {
                String currentIpSegment = s.substring(begin, i + 1);
                path.addLast(currentIpSegment);

                restoreIpAddressesDfs(s, i + 1, residue - 1, path, res);
                path.removeLast();
            }
        }
    }

    private boolean restoreIpAddressesJudgeIpSegment(String s, int left, int right) {
        int len = right - left + 1;
        // 开头为 0 的，并且长度大于 1 的数字需要进行剪枝
        if (len > 1 && s.charAt(left) == '0') {
            return false;
        }
        int res = 0;
        while (left <= right) {
            res = res * 10 + s.charAt(left) - '0';
            left++;
        }
        return res >= 0 && res <= 255;
    }

    // -------复原IP地址 << end --------
}
