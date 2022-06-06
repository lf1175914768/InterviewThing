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

    // -------Z字形变换 start >>--------

    /**
     * 将一个给定字符串 s 根据给定的行数 numRows ，以从上往下、从左到右进行 Z 字形排列。
     * 比如输入字符串为 "PAYPALISHIRING" 行数为 3 时，排列如下：
     * P   A   H   N
     * A P L S I I G
     * Y   I   R
     * 之后，你的输出需要从左往右逐行读取，产生出一个新的字符串，比如："PAHNAPLSIIGYIR"。
     *
     * 对应 leetcode 中第 6 题。
     */
    public String convert(String s, int numRows) {
        if (numRows < 2) return s;
        List<StringBuilder> rows = new ArrayList<>();
        for (int i = 0; i < numRows; i++) {
            rows.add(new StringBuilder());
        }
        int i = 0, flag = -1;
        for (char c : s.toCharArray()) {
            rows.get(i).append(c);
            if (i == 0 || i == numRows - 1) flag = -flag;
            i += flag;
        }
        StringBuilder res = new StringBuilder();
        for (StringBuilder row : rows) {
            res.append(row);
        }
        return res.toString();
    }

    // -------Z字形变换 << end --------

    // -------替换后的最长重复字符 start >>--------

    /**
     * 给你一个字符串 s 和一个整数 k 。你可以选择字符串中的任一字符，并将其更改为任何其他大写英文字符。该操作最多可执行 k 次。
     * 在执行上述操作后，返回包含相同字母的最长子字符串的长度。
     *
     * 使用 滑动窗口的方法进行解答。
     *
     * 对应 leetcode 中第 424 题。
     */
    public int characterReplacement(String s, int k) {
        int left = 0, right = 0, len = s.length(), maxCount = 0;
        int res = 0;
        Map<Character, Integer> window = new HashMap<>();
        while (right < len) {
            char c = s.charAt(right);
            window.put(c, window.getOrDefault(c, 0) + 1);
            maxCount = Math.max(maxCount, window.get(c));
            right++;
            while (right - left > maxCount + k) {
                char ch = s.charAt(left);
                window.put(ch, window.get(ch) - 1);
                left++;
            }
            res = Math.max(res, right - left);
        }
        return res;
    }

    // -------替换后的最长重复字符 << end --------

    // -------串联所有单词的子串 start >>--------

    /**
     * 给定一个字符串 s 和一些 长度相同 的单词 words 。找出 s 中恰好可以由 words 中所有单词串联形成的子串的起始位置。
     * 注意子串要与 words 中的单词完全匹配，中间不能有其他字符 ，但不需要考虑 words 中单词串联的顺序。
     *
     * 使用滑动窗口的方法进行解答。
     *
     * 对应 leetcode 中第 30 题。
     */
    public List<Integer> findSubstring(String s, String[] words) {
        List<Integer> res = new ArrayList<>();
        int oneLength = words[0].length();
        Map<String, Integer> map = new HashMap<>();
        for (String word : words) {
            map.put(word, map.getOrDefault(word, 0) + 1);
        }
        Map<String, Integer> work = new HashMap<>();
        for (int i = 0; i < oneLength; i++) {
            int left = i, right = i + oneLength;
            int count = 0;
            // 这里使用小于等于，是因为substring，区间右侧是开区间
            while (right <= s.length()) {
                String cur = s.substring(right - oneLength, right);
                if (!map.containsKey(cur)) {
                    // map 中不包含当前截取的字符串，直接向下滑动
                    left = right;
                    count = 0;
                    work.clear();
                } else {
                    work.put(cur, work.getOrDefault(cur, 0) + 1);
                    count++;
                    while (work.get(cur) > map.get(cur)) {
                        String leftStr = s.substring(left, left + oneLength);
                        work.put(leftStr, work.get(leftStr) - 1);
                        count--;
                        left += oneLength;
                    }
                    if (count == words.length) res.add(left);
                }
                right += oneLength;
            }
            work.clear();
        }
        return res;
    }

    // -------串联所有单词的子串 << end --------

    // -------整数转罗马数字 start >>--------

    /**
     * 罗马数字包含以下七种字符： I， V， X， L，C，D 和 M。
     * 例如， 罗马数字 2 写做 II ，即为两个并列的 1。12 写做 XII ，即为 X + II 。 27 写做  XXVII, 即为 XX + V + II 。
     *
     * 通常情况下，罗马数字中小的数字在大的数字的右边。但也存在特例，例如 4 不写做 IIII，而是 IV。数字 1 在数字 5 的左边，
     * 所表示的数等于大数 5 减小数 1 得到的数值 4 。同样地，数字 9 表示为 IX。这个特殊的规则只适用于以下六种情况：
     *
     * I 可以放在 V (5) 和 X (10) 的左边，来表示 4 和 9。
     * X 可以放在 L (50) 和 C (100) 的左边，来表示 40 和 90。 
     * C 可以放在 D (500) 和 M (1000) 的左边，来表示 400 和 900。
     * 给你一个整数，将其转为罗马数字。
     *
     * 使用贪心的算法思想进行解答。
     *
     * 对应 leetcode 中第 12 题。
     */
    public String intToRoman(int num) {
        int[] nums = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
        String[] romans = {"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"};

        StringBuilder sb = new StringBuilder();
        int index = 0;
        while (index < 13) {
            while (num >= nums[index]) {
                sb.append(romans[index]);
                num -= nums[index];
            }
            index++;
        }
        return sb.toString();
    }

    // -------整数转罗马数字 << end --------

    // -------字符串相乘 start >>--------

    /**
     * 给定两个以字符串形式表示的非负整数 num1 和 num2，返回 num1 和 num2 的乘积，它们的乘积也表示为字符串形式。
     * 注意：不能使用任何内置的 BigInteger 库或直接将输入转换为整数。
     *
     * 自己写的，有可能有疏漏的地方。
     *
     * 对应 leetcode 中第 43 题。
     */
    public String multiply(String num1, String num2) {
        int res = 0, rax = 0;
        char[] chars = num2.toCharArray();
        for (int i = chars.length - 1; i >= 0; i--) {
            char c = chars[i];
            int curMulti = multiplyToCalc(num1, c);
            res += curMulti * Math.pow(10, rax++);
        }
        return String.valueOf(res);
    }

    private int multiplyToCalc(String num1, char c) {
        int prev = 0, res = 0, rax = 0;
        char[] chars = num1.toCharArray();
        for (int i = chars.length - 1; i >= 0; i--) {
            char ch = chars[i];
            int tmp = (c - '0') * (ch - '0') + prev;
            prev = tmp / 10;
            res += (tmp % 10) * Math.pow(10, rax++);
        }
        return res;
    }

    public String multiply_v2(String num1, String num2) {
        if (num1.equals("0") || num2.equals("0")) return "0";
        int[] res = new int[num1.length() + num2.length()];
        for (int i = num1.length() - 1; i >= 0; i--) {
            int n1 = num1.charAt(i) - '0';
            for (int j = num2.length() - 1; j >= 0; j--) {
                int n2 = num2.charAt(j) - '0';
                int sum = res[i + j + 1] + n1 * n2;
                res[i + j + 1] = sum % 10;
                res[i + j] += sum / 10;
            }
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < res.length; i++) {
            if (i == 0 && res[i] == 0) continue;
            sb.append(res[i]);
        }
        return sb.toString();
    }

    // -------字符串相乘 << end --------

    // -------文本左右对齐 start >>--------

    /**
     * 给定一个单词数组 words 和一个长度 maxWidth ，重新排版单词，使其成为每行恰好有 maxWidth 个字符，且左右两端对齐的文本。
     * 你应该使用 “贪心算法” 来放置给定的单词；也就是说，尽可能多地往每行中放置单词。必要时可用空格 ' ' 填充，使得每行恰好有 maxWidth 个字符。
     * 要求尽可能均匀分配单词间的空格数量。如果某一行单词间的空格不能均匀分配，则左侧放置的空格数要多于右侧的空格数。
     * 文本的最后一行应为左对齐，且单词之间不插入额外的空格。
     *
     * 注意:
     * 单词是指由非空格字符组成的字符序列。
     * 每个单词的长度大于 0，小于等于 maxWidth。
     * 输入单词数组 words 至少包含一个单词。
     *
     * 对应 leetcode 中第 68 题。
     */
    public List<String> fullJustify(String[] words, int maxWidth) {
        List<String> res = new ArrayList<>();
        int cnt = 0, start = 0;
        for (int i = 0; i < words.length; i++) {
            cnt += words[i].length() + 1;
            // 如果是最后一个单词，或者加上下一单词就超过长度了，即可凑成一行
            if (i + 1 == words.length || cnt + words[i + 1].length() > maxWidth) {
                res.add(fullJustifyFillWords(words, start, i, maxWidth, i + 1 == words.length));
                start = i + 1;
                cnt = 0;
            }
        }
        return res;
    }

    private String fullJustifyFillWords(String[] words, int start, int end, int maxWidth, boolean lastLine) {
        int wordCount = end - start + 1;
        // 除去每个单词尾部空格。
        int spaceCount = maxWidth - wordCount + 1;
        for (int i = start; i <= end; i++) {
            // 除去所有单词的长度
            spaceCount -= words[i].length();
        }
        // 词尾空格
        final int spaceSuffix = 1;
        // 额外空格的平均值 = 总空格数/间隙数
        final int spaceAvg = wordCount == 1 ? 0 : spaceCount / (wordCount - 1);
        int spaceExtra = wordCount == 1 ? 0 : spaceCount % (wordCount - 1);
        StringBuilder sb = new StringBuilder();
        for (int i = start; i < end; i++) {
            sb.append(words[i]);
            if (lastLine) {
                sb.append(" ");
                continue;
            }
            int n = spaceSuffix + spaceAvg + (spaceExtra-- > 0 ? 1 : 0);
            while (n-- > 0) {
                sb.append(" ");
            }
        }
        // 插入最后一个单词
        sb.append(words[end]);
        int lastSpaceCnt = maxWidth - sb.length();
        while (lastSpaceCnt-- > 0) {
            sb.append(" ");
        }
        return sb.toString();
    }

    // -------文本左右对齐 << end --------

    // -------简化路径 start >>--------

    /**
     * 给你一个字符串 path ，表示指向某一文件或目录的 Unix 风格 绝对路径 （以 '/' 开头），请你将其转化为更加简洁的规范路径。
     * 在 Unix 风格的文件系统中，一个点（.）表示当前目录本身；此外，两个点 （..） 表示将目录切换到上一级（指向父目录）；
     * 两者都可以是复杂相对路径的组成部分。任意多个连续的斜杠（即，'//'）都被视为单个斜杠 '/' 。 对于此问题，任何其他格式的点（例如，'...'）均被视为文件/目录名称。
     *
     * 请注意，返回的 规范路径 必须遵循下述格式：
     * 始终以斜杠 '/' 开头。
     * 两个目录名之间必须只有一个斜杠 '/' 。
     * 最后一个目录名（如果存在）不能 以 '/' 结尾。
     * 此外，路径仅包含从根目录到目标文件或目录的路径上的目录（即，不含 '.' 或 '..'）。
     * 返回简化后得到的 规范路径 。
     *
     * 对应 leetcode 中第 71 题。
     */
    public String simplifyPath(String path) {
        Deque<String> queue = new LinkedList<>();
        String[] names = path.split("/");
        for (String name : names) {
            if ("..".equals(name)) {
                if (!queue.isEmpty())
                    queue.pollLast();
            } else if (name.length() > 0 && !".".equals(name)) {
                queue.offerLast(name);
            }
        }
        StringBuilder sb = new StringBuilder();
        while (!queue.isEmpty()) {
            sb.append("/").append(queue.pollFirst());
        }
        return sb.length() == 0 ? "/" : sb.toString();
    }

    public String simplifyPath_v2(String path) {
        Deque<String> queue = new LinkedList<>();
        for (int i = 1; i < path.length();) {
            if (path.charAt(i) == '/') {
                i++;
                continue;
            }
            int j = i + 1;
            while (j < path.length() && path.charAt(j) != '/') j++;
            String item = path.substring(i, j);
            if (item.equals("..")) {
                if (!queue.isEmpty())
                    queue.pollLast();
            } else if (!item.equals("."))
                queue.offerLast(item);
            i = j;
        }
        StringBuilder sb = new StringBuilder();
        while (!queue.isEmpty()) {
            sb.append("/").append(queue.pollFirst());
        }
        return sb.length() == 0 ? "/" : sb.toString();
    }

    // -------简化路径 << end --------

    // -------单词接龙II start >>--------

    /**
     * 按字典 wordList 完成从单词 beginWord 到单词 endWord 转化，一个表示此过程的 转换序列 是形式上像
     * beginWord -> s1 -> s2 -> ... -> sk 这样的单词序列，并满足：
     * 1.每对相邻的单词之间仅有单个字母不同。
     * 2.转换过程中的每个单词 si（1 <= i <= k）必须是字典 wordList 中的单词。注意，beginWord 不必是字典 wordList 中的单词。
     * 3.sk == endWord
     * 给你两个单词 beginWord 和 endWord ，以及一个字典 wordList 。请你找出并返回所有从 beginWord 到 endWord 的 最短转换序列 ，
     * 如果不存在这样的转换序列，返回一个空列表。每个序列都应该以单词列表 [beginWord, s1, s2, ..., sk] 的形式返回。
     *
     * 采用 BFS 的方法进行解答。
     *
     * 对应 leetcode 中第 126 题。
     */
    public List<List<String>> findLadders(String beginWord, String endWord, List<String> wordList) {
        List<List<String>> res = new ArrayList<>();
        // 如果不含有结束单词，直接结束，不然后边会造成死循环
        if (!wordList.contains(endWord)) {
            return res;
        }
        Queue<List<String>> queue = new LinkedList<>();
        List<String> path = new ArrayList<>();
        path.add(beginWord);
        queue.offer(path);
        Set<String> dict = new HashSet<>(wordList);
        Set<String> visited = new HashSet<>();
        visited.add(beginWord);
        boolean found = false;
        while (!queue.isEmpty()) {
            int size = queue.size();
            Set<String> subVisited = new HashSet<>();
            for (int i = 0; i < size; i++) {
                List<String> p = queue.poll();
                // 得到当前路径的末尾单词
                String temp = p.get(p.size() - 1);
                // 一次性得到所有的下一个的节点
                List<String> neighbors = findLadderGetNeighbors(temp, dict);
                for (String neighbor : neighbors) {
                    // 只考虑之前没有出现过的单词
                    if (!visited.contains(neighbor)) {
                        if (neighbor.equals(endWord)) {
                            found = true;
                            p.add(neighbor);
                            res.add(new ArrayList<>(p));
                            p.remove(p.size() - 1);
                        }
                        p.add(neighbor);
                        queue.offer(new ArrayList<>(p));
                        p.remove(p.size() - 1);
                        subVisited.add(neighbor);
                    }
                }
            }
            visited.addAll(subVisited);
            if (found) break;
        }
        return res;
    }

    private List<String> findLadderGetNeighbors(String node, Set<String> dict) {
        List<String> res = new ArrayList<>();
        char[] chs = node.toCharArray();
        for (char ch = 'a'; ch <= 'z'; ch++) {
            for (int i = 0; i < node.length(); i++) {
                if (node.charAt(i) == ch) {
                    continue;
                }
                char oldCh = chs[i];
                chs[i] = ch;
                if (dict.contains(String.valueOf(chs))) {
                    res.add(String.valueOf(chs));
                }
                chs[i] = oldCh;
            }
        }
        return res;
    }

    // -------单词接龙II << end --------

    // -------去除重复字母 start >>--------

    /**
     * 给你一个字符串 s ，请你去除字符串中重复的字母，使得每个字母只出现一次。需保证 返回结果的字典序最小（要求不能打乱其他字符的相对位置）。
     *
     * 首先考虑一个简单的问题：给定一个字符串s，如何去掉其中的一个字符 ch，使得得到的字符串字典序最小？
     * 答案是：找出最小的满足 s[i] > s[i + 1] 的下标 i，并去除字符 s[i].为了叙述方便，下文中称这样的字符为 【关键字符】.
     * 我们从前往后扫描原字符串，每扫描到一个位置，我们就尽可能的处理所有的 【关键字符】。假定在扫描位置 s[i - 1]之前的所有的【关键字福】都已经被去除完毕。
     * 在扫描字符 s[i]时，新出现的【关键字符】只可能出现在 s[i]或者其后面的位置。
     * 于是，我们使用单调栈来维护去除【关键字符】后得到的字符串，单调栈满足栈底到栈顶的字符递增。如果栈顶字符大于当前字符 s[i]，说明栈顶字符为【关键字符】，
     * 故应当被去除。去除后，新的栈顶字符就与 s[i]相邻了，我们继续比较新的栈顶字符与 s[i] 的大小。重复上述操作，知道栈为空或者栈顶字符不大于 s[i]。
     *
     * 我们还遗漏了一个要求：原字符串 s 中的每个字符都需要出现在新字符串中，且只能出现一次。为了让新字符串满足该要求，之前讨论的算法需要进行以下两点的更改。
     * 1、在考虑字符 s[i]时，如果它已经存在于栈中，则不能加入字符 s[i]。为此，需要记录每个字符是否出现在栈中。
     * 2、在弹出栈顶字符时，如果字符串在后面的位置上再也没有这一字符，则不能弹出栈顶字符。为此，需要记录每个字符的剩余数量，当这个值为 0时，
     *    就不能再弹出栈顶字符了。
     *
     * 对应 leetcode 中第 316 题。
     */
    public String removeDuplicateLetters(String s) {
        Map<Character, Integer> map = new HashMap<>();
        for (char c : s.toCharArray()) {
            map.put(c, map.getOrDefault(c, 0) + 1);
        }
        boolean[] used = new boolean[26];
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            // 如果当前字符没有出现过
            if (!used[ch - 'a']) {
                while (sb.length() > 0 && sb.charAt(sb.length() - 1) > ch) {
                    char headC = sb.charAt(sb.length() - 1);
                    if (map.get(headC) > 0) {
                        used[headC - 'a'] = false;
                        sb.deleteCharAt(sb.length() - 1);
                    } else {
                        break;
                    }
                }
                used[ch - 'a'] = true;
                sb.append(ch);
            }
            map.put(ch, map.get(ch) - 1);
        }
        return sb.toString();
    }

    // -------去除重复字母 << end --------

    // -------基本计算器 start >>--------

    /**
     * 给你一个字符串表达式 s ，请你实现一个基本计算器来计算并返回它的值。
     * 注意:不允许使用任何将字符串作为数学表达式计算的内置函数，比如 eval() 。
     *
     * 对应  leetcode 中第 224 题。
     */
    public int calculate(String s) {
        // 存放所有的数字
        Deque<Integer> nums = new ArrayDeque<>();
        // 为了防止第一个数为负数，先往nums中加一个 0
        nums.addLast(0);
        s = s.replaceAll("\\s", "");
        // 存放所有的操作，包括 +/-
        Deque<Character> ops = new ArrayDeque<>();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '(') {
                ops.addLast(c);
            } else if (c == ')') {
                // 计算直到最近一个左括号为止
                while (!ops.isEmpty()) {
                    Character op = ops.peekLast();
                    if (op != '(') {
                        calculate(nums, ops);
                    } else {
                        ops.pollLast();
                        break;
                    }
                }
            } else {
                if (Character.isDigit(c)) {
                    int u = 0, j = i;
                    while (j < s.length() && Character.isDigit(s.charAt(j))) {
                        u = u * 10 + (s.charAt(j++) - '0');
                    }
                    nums.add(u);
                    i = j - 1;
                } else {
                    char preC;
                    if (i > 0 && ((preC = s.charAt(i - 1)) == '('
                                || preC == '+' || preC == '-')) {
                        nums.add(0);
                    }
                    // 当有一个新操作要入栈时，先把栈内可以算的都算了
                    while (!ops.isEmpty() && ops.peekLast() != '(') {
                        calculate(nums, ops);
                    }
                    ops.addLast(c);
                }
            }
        }
        while (!ops.isEmpty()) {
            calculate(nums, ops);
        }
        return nums.peekLast();
    }

    private void calculate(Deque<Integer> nums, Deque<Character> ops) {
        if (nums.isEmpty() || nums.size() < 2 || ops.isEmpty()) return;
        int b = nums.pollLast(), a = nums.pollLast();
        char op = ops.pollLast();
        nums.addLast(op == '+' ? a + b : a - b);
    }

    // -------基本计算器 << end --------

    // -------组合总和 start >>--------

    public List<List<Integer>> permute1(int[] nums) {
        return null;
    }

    // -------组合总和 << end --------
}
