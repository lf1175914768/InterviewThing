package com.study.leetcode;

import java.util.Arrays;
import java.util.List;

/**
 * <p>description: top 面试题  </p>
 * <p>className:  TopInterviewProblems </p>
 * <p>create time:  2022/3/9 14:28 </p>
 *
 * @author feng.liu
 * @since 1.0
 **/
public class TopInterviewProblems {

    // -------字符串转换整数 start >>--------

    /**
     * 请你来实现一个 myAtoi(string s) 函数，使其能将字符串转换成一个 32 位有符号整数（类似 C/C++ 中的 atoi 函数）。
     * <p>
     * 函数 myAtoi(string s) 的算法如下：
     * <p>
     * 读入字符串并丢弃无用的前导空格
     * 检查下一个字符（假设还未到字符末尾）为正还是负号，读取该字符（如果有）。 确定最终结果是负数还是正数。 如果两者都不存在，则假定结果为正。
     * 读入下一个字符，直到到达下一个非数字字符或到达输入的结尾。字符串的其余部分将被忽略。
     * 将前面步骤读入的这些数字转换为整数（即，"123" -> 123， "0032" -> 32）。如果没有读入数字，则整数为 0 。必要时更改符号（从步骤 2 开始）。
     * 如果整数数超过 32 位有符号整数范围 [−231,  231 − 1] ，需要截断这个整数，使其保持在这个范围内。具体来说，小于 −231 的整数应该被固定为 −231 ，大于 231 − 1 的整数应该被固定为 231 − 1 。
     * 返回整数作为最终结果。
     * 注意：
     * <p>
     * 本题中的空白字符只包括空格字符 ' ' 。
     * 除前导空格或数字后的其余字符串外，请勿忽略 任何其他字符。
     * <p>
     * 对应 leetcode 中第 8 题。
     */
    public int myAtoi(String s) {
        int res = 0;
        char[] arr = s.toCharArray();
        int index = 0;
        while (index < arr.length && arr[index] == ' ')
            index++;
        if (index == arr.length)
            return res;
        int sign = 1;
        if (arr[index] == '-') {
            sign = -1;
            index++;
        } else if (arr[index] == '+')
            index++;
        int maxValue = Integer.MAX_VALUE / 10, maxValueLeft = Integer.MAX_VALUE % 10;
        int minValue = Integer.MIN_VALUE / 10, minValueLeft = -(Integer.MIN_VALUE % 10);
        while (index < arr.length) {
            char c = arr[index];
            if (!Character.isDigit(c)) {
                break;
            }
            if (res > maxValue || (res == maxValue && (c - '0') > maxValueLeft)) {
                return Integer.MAX_VALUE;
            }
            if (res < minValue || (res == minValue && (c - '0') > minValueLeft)) {
                return Integer.MIN_VALUE;
            }
            res = res * 10 + sign * (c - '0');
            index++;
        }
        return res;
    }

    // -------字符串转换整数 << end --------

    // -------零钱兑换 start >>--------

    /**
     * 给你一个整数数组 coins ，表示不同面额的硬币；以及一个整数 amount ，表示总金额。
     * 计算并返回可以凑成总金额所需的 最少的硬币个数 。如果没有任何一种硬币组合能组成总金额，返回 -1 。
     * 你可以认为每种硬币的数量是无限的。
     * <p>
     * 使用 递归的方式进行实现。
     * <p>
     * 对应 leetcode 中第 322 题。
     */
    public int coinChange(int[] coins, int amount) {
        if (coins.length == 0)
            return -1;
        int[] memo = new int[amount];
        return coinChangeFindWay(coins, amount, memo);
    }

    // memo[n] 表示钱币 n 可以被换取的最少的钱币数，不能换取就为 -1
    private int coinChangeFindWay(int[] coins, int amount, int[] memo) {
        if (amount < 0) {
            return -1;
        }
        if (amount == 0) {
            return 0;
        }
        if (memo[amount - 1] != 0) {
            return memo[amount - 1];
        }
        int min = Integer.MAX_VALUE;
        for (int coin : coins) {
            int res = coinChangeFindWay(coins, amount - coin, memo);
            if (res >= 0 && res < min) {
                min = res + 1;
            }
        }
        memo[amount - 1] = (min == Integer.MAX_VALUE ? -1 : min);
        return memo[amount - 1];
    }

    /**
     * 上面的动态规划 版本
     */
    public int coinChange_v2(int[] coins, int amount) {
        if (coins.length == 0)
            return -1;

        // memo[n] 的值，表示凑成总金额为n所需的最少的硬币个数
        int[] memo = new int[amount + 1];
        memo[0] = 0;
        for (int i = 1; i <= amount; i++) {
            int min = Integer.MAX_VALUE;
            for (int coin : coins) {
                if (i - coin >= 0 && memo[i - coin] < min) {
                    // 这里一定要 +1， 表示在前面的基础上，需要在挑选一个
                    min = memo[i - coin] + 1;
                }
            }
            memo[i] = min;
        }
        return memo[amount] == Integer.MAX_VALUE ? -1 : memo[amount];
    }

    // -------零钱兑换 << end --------

    // -------被围绕的区域 start >>--------

    /**
     * 给你一个 m x n 的矩阵 board ，由若干字符 'X' 和 'O' ，找到所有被 'X' 围绕的区域，并将这些区域里所有的 'O' 用 'X' 填充。
     * <p>
     * 对应 leetcode 中第 130 题
     */
    public void solve(char[][] board) {
        if (board == null || board.length == 0)
            return;
        int m = board.length;
        int n = board[0].length;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                boolean isEdge = i == 0 || i == m - 1 || j == 0 || j == n - 1;
                if (isEdge && board[i][j] == '0') {
                    solveDfs(board, i, j);
                }
            }
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (board[i][j] == '0') {
                    board[i][j] = 'X';
                } else if (board[i][j] == '#') {
                    board[i][j] = '0';
                }
            }
        }
    }

    private void solveDfs(char[][] board, int i, int j) {
        if (i < 0 || j < 0 || i >= board.length || j >= board[0].length || board[i][j] == 'X' || board[i][j] == '#')
            return;
        board[i][j] = '#';
        solveDfs(board, i - 1, j);
        solveDfs(board, i + 1, j);
        solveDfs(board, i, j - 1);
        solveDfs(board, i, j + 1);
    }

    // -------被围绕的区域 << end --------

    // -------最大数 start >>--------

    /**
     * 给定一组非负整数 nums，重新排列每个数的顺序（每个数不可拆分）使之组成一个最大的整数。
     * 注意：输出结果可能非常大，所以你需要返回一个字符串而不是整数。
     * <p>
     * 对应 leetcode 中第 179 题。
     */
    public String largestNumber(int[] nums) {
        int n = nums.length;
        String[] ss = new String[n];
        for (int i = 0; i < n; i++) {
            ss[i] = "" + nums[i];
        }
        Arrays.sort(ss, (a, b) -> {
            String sa = a + b, sb = b + a;
            return sb.compareTo(sa);
        });

        StringBuilder sb = new StringBuilder();
        for (String s : ss) {
            sb.append(s);
        }
        int k = 0;
        while (k < sb.length() - 1 && sb.charAt(k) == '0') k++;
        return sb.substring(k);
    }

    // -------最大数 << end --------

    // -------排序链表 start >>--------

    /**
     * 给你链表的头结点 head ，请将其按 升序 排列并返回 排序后的链表 。
     *
     * 通过递归实现链表归并排序，有以下两个环节：
     * 1. 分割cut环节：找到当前链表中点，并从中点将链表断开（以便在下次递归cut时，链表片段拥有正确边界）；
     *    我们使用 fast，slow 快慢双指针法，奇数个节点找到中点，偶数个节点找到中心左边的节点。
     *    找到中点 slow 后，执行 slow.next = None 将链表切断。
     *    递归分割时，输入当前链表左端点 head 和 中心节点 slow 的下一个节点 tmp （因为链表是从 slow 切断的。）
     *    cut 递归终止条件：当 head.next = None 时，说明只有一个节点了，直接返回此节点。
     * 2. 合并 merge 节点：将两个排序链表合并，转化为一个排序链表。
     *    双指针合并，建立辅助 ListNode h 作为头部。
     *    设置两指针 left，right 分别指向两链表头部，比较两指针处节点值大小，由小到大加入合并链表头部，指针交替前进，直至添加完两个链表。
     *    返回辅助 ListNode h 作为头部的下个节点 h.next.
     *    时间复杂度 o(l + r), l，r 分别代表两个链表长度。
     * 当题目输入的 head == None 时， 直接返回 None。
     *
     * 对应 leetcode 中第 148 题。
     */
    public ListNode sortList(ListNode head) {
        if (head == null || head.next == null)
            return head;
        ListNode fast = head.next, slow = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode tmp = slow.next;
        slow.next = null;
        ListNode left = sortList(head);
        ListNode right = sortList(tmp);
        ListNode h = new ListNode(0);
        ListNode res = h;
        while (left != null && right != null) {
            if (left.val < right.val) {
                h.next = left;
                left = left.next;
            } else {
                h.next = right;
                right = right.next;
            }
            h = h.next;
        }
        h.next = left != null ? left : right;
        return res.next;
    }

    // -------排序链表 << end --------

    // -------寻找峰值 start >>--------

    /**
     * 峰值元素是指其值严格大于左右相邻值的元素。
     * 给你一个整数数组 nums，找到峰值元素并返回其索引。数组可能包含多个峰值，在这种情况下，返回 任何一个峰值 所在位置即可。
     * 你可以假设 nums[-1] = nums[n] = -∞ 。
     * 你必须实现时间复杂度为 O(log n) 的算法来解决此问题。
     *
     * 对应 leetcode 中第 162 题。
     */
    public int findPeakElement(int[] nums) {
        int left = 0, right = nums.length - 1;
        while (left < right) {
            int middle = left + ((right - left) >>> 1);
            if (nums[middle] > nums[middle + 1]) {
                right = middle;
            } else {
                left = middle + 1;
            }
        }
        return left;
    }

    // -------寻找峰值 << end --------
}
