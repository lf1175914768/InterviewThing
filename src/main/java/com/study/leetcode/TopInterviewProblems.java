package com.study.leetcode;

import java.util.*;

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

    // -------分割回文串 start >>--------

    /**
     * 给你一个字符串 s，请你将 s 分割成一些子串，使每个子串都是 回文串 。返回 s 所有可能的分割方案。
     * 回文串 是正着读和反着读都一样的字符串。
     *
     * 使用回溯算法进行实现。
     *
     * 对应 leetcode 中第 131 题。
     */
    public List<List<String>> partition(String s) {
        List<List<String>> result = new ArrayList<>();
        if (null == s || s.length() == 0) {
            return result;
        }
        char[] arr = s.toCharArray();
        Deque<String> path = new LinkedList<>();
        partitionDfs(arr, 0, s.length(), path, result);
        return result;
    }

    private void partitionDfs(char[] arr, int index, int end, Deque<String> path, List<List<String>> result) {
        if (index == end) {
            result.add(new ArrayList<>(path));
            return;
        }
        for (int i = index; i < end; i++) {
            if (!partitionCheckPass(arr, index, i)) {
                continue;
            }
            path.offerLast(String.valueOf(arr, index, i - index + 1));
            partitionDfs(arr, i + 1, end, path, result);
            path.removeLast();
        }
    }

    private boolean partitionCheckPass(char[] arr, int start, int end) {
        while (start < end) {
            if (arr[start] != arr[end]) {
                return false;
            }
            start++;
            end--;
        }
        return true;
    }

    /**
     * 使用动态规划进行处理
     */
    public List<List<String>> partition_v2(String s) {
        List<List<String>> result = new ArrayList<>();
        if (null == s || s.length() == 0) {
            return result;
        }
        boolean[][] dp = new boolean[s.length()][s.length()];
        for (int right = 0; right < s.length(); right++) {
            for (int left = right; left >= 0; left--) {
                if (s.charAt(left) == s.charAt(right) && (right - left <= 2 || dp[left + 1][right - 1])) {
                    dp[left][right] = true;
                }
            }
        }
        Deque<String> stack = new ArrayDeque<>();
        partitionV2Dfs(s, 0, s.length(), dp, stack, result);
        return result;
    }

    private void partitionV2Dfs(String s, int index, int length, boolean[][] dp, Deque<String> stack, List<List<String>> result) {
        if (index == length) {
            result.add(new ArrayList<>(stack));
            return;
        }
        for (int i = index; i < length; i++) {
            if (dp[index][i]) {
                stack.addLast(s.substring(index, i + 1));
                partitionV2Dfs(s, i + 1, length, dp, stack, result);
                stack.removeLast();
            }
        }
    }

    // -------分割回文串 << end --------

    // -------复制带随机指针的链表 start >>--------

    /**
     * 给你一个长度为 n 的链表，每个节点包含一个额外增加的随机指针 random ，该指针可以指向链表中的任何节点或空节点。
     * 构造这个链表的 深拷贝。 深拷贝应该正好由 n 个 全新 节点组成，其中每个新节点的值都设为其对应的原节点的值。新节点的 next 指针和
     * random 指针也都应指向复制链表中的新节点，并使原链表和复制链表中的这些指针能够表示相同的链表状态。复制链表中的指针都不应指向原链表中的节点 。
     * 例如，如果原链表中有 X 和 Y 两个节点，其中 X.random --> Y 。那么在复制链表中对应的两个节点 x 和 y ，同样有 x.random --> y 。
     * 返回复制链表的头节点。
     * 用一个由 n 个节点组成的链表来表示输入/输出中的链表。每个节点用一个 [val, random_index] 表示：
     *
     * val：一个表示 Node.val 的整数。
     * random_index：随机指针指向的节点索引（范围从 0 到 n-1）；如果不指向任何节点，则为  null 。
     * 你的代码 只 接受原链表的头节点 head 作为传入参数。
     *
     * 对应 leetcode 中第 138 题。
     */
    public RandomNode copyRandomList(RandomNode head) {
        for (RandomNode p = head; p != null; p = p.next.next) {
            RandomNode q = new RandomNode(p.val);
            q.next = p.next;
            p.next = q;
        }

        for (RandomNode p = head; p != null; p = p.next.next) {
            // 复制 random 指针
            if (p.random != null)
                p.next.random = p.random.next;
        }

        // 拆分两个链表，并复原原链表
        RandomNode dummy = new RandomNode(-1), cur = dummy;
        for (RandomNode p = head; p != null; p = p.next) {
            RandomNode q = p.next;
            cur.next = q;
            cur = cur.next;
            p.next = q.next;
        }
        return dummy.next;
    }

    /**
     * 回溯算法的实现
     */
    public RandomNode copyRandomList_v2(RandomNode head) {
        if (head == null)
            return null;
        Map<RandomNode, RandomNode> hash = new HashMap<>();
        return copyRandomListDfs(head, hash);
    }

    private RandomNode copyRandomListDfs(RandomNode node, Map<RandomNode, RandomNode> hash) {
        if (node == null) {
            return null;
        }
        if (hash.containsKey(node))
            return hash.get(node);
        RandomNode clone = new RandomNode(node.val);  // 复制节点
        hash.put(node, clone);    // 建立源节点到复制节点的映射
        clone.next = copyRandomListDfs(node.next, hash);
        clone.random = copyRandomListDfs(node.random, hash);
        return clone;
    }

    // -------复制带随机指针的链表 << end --------

    // -------轮转数组 start >>--------

    /**
     * 给你一个数组，将数组中的元素向右轮转 k 个位置，其中 k 是非负数。
     *
     * 对应 leetcode 中第 189 题。
     */
    public void rotate(int[] nums, int k) {
        k %= nums.length;
        rotateReverse(nums, 0, nums.length - 1);
        rotateReverse(nums, 0, k - 1);
        rotateReverse(nums, k, nums.length - 1);
    }

    private void rotateReverse(int[] nums, int start, int end) {
        while (start < end) {
            int tmp = nums[start];
            nums[start] = nums[end];
            nums[end] = tmp;
            start++;
            end--;
        }
    }


    // -------轮转数组 << end --------

    // -------奇偶链表 start >>--------

    /**
     * 给定单链表的头节点 head ，将所有索引为奇数的节点和索引为偶数的节点分别组合在一起，然后返回重新排序的列表。
     * 第一个节点的索引被认为是 奇数 ， 第二个节点的索引为 偶数 ，以此类推。
     * 请注意，偶数组和奇数组内部的相对顺序应该与输入时保持一致。
     * 你必须在 O(1) 的额外空间复杂度和 O(n) 的时间复杂度下解决这个问题。
     *
     * 对应 leetcode 中第328 题。
     */
    public ListNode oddEvenList(ListNode head) {
        ListNode dummy = new ListNode(0), cur = dummy, pre = dummy;
        for (ListNode p = head; p != null; p = p.next) {
            ListNode node = p.next;
            cur.next = node;
            cur = cur.next;
            pre = p;
            if (node != null) {
                p.next = node.next;
            }
        }
        pre.next = dummy.next;
        return head;
    }

    // -------奇偶链表 << end --------

    // -------递增的三元子序列 start >>--------

    /**
     * 给你一个整数数组 nums ，判断这个数组中是否存在长度为 3 的递增子序列。
     * 如果存在这样的三元组下标 (i, j, k) 且满足 i < j < k ，使得 nums[i] < nums[j] < nums[k] ，返回 true ；否则，返回 false 。
     *
     * 对应 leetcode 中第 334 题。
     */
    public boolean increasingTriplet(int[] nums) {
        int n = nums.length;
        if (n < 3)
            return false;
        int first = nums[0], second = Integer.MAX_VALUE;
        for (int i = 1; i < n; i++) {
            int num = nums[i];
            if (num > second) {
                return true;
            } else if (num > first) {
                second = num;
            } else {
                first = num;
            }
        }
        return false;
    }

    // -------递增的三元子序列 << end --------

    // -------两整数之和 start >>--------

    /**
     * 给你两个整数 a 和 b ，不使用 运算符 + 和 - ​​​​​​​，计算并返回两整数之和。
     *
     * 对应 leetcode 中第 371 题。
     */
    public int getSum(int a, int b) {
        while (b != 0) {
            int carry = (a & b) << 1;
            a = a ^ b;
            b = carry;
        }
        return a;
    }

    // -------两整数之和 << end --------

    // -------有序矩阵中第K小的元素 start >>--------

    /**
     * 给你一个 n x n 矩阵 matrix ，其中每行和每列元素均按升序排序，找到矩阵中第 k 小的元素。
     * 请注意，它是 排序后 的第 k 小元素，而不是第 k 个 不同 的元素。
     * 你必须找到一个内存复杂度优于 O(n2) 的解决方案。
     *
     * 利用小根堆 归并排序。
     *
     * 对应 leetcode 中第 378 题。
     */
    public int kthSmallest(int[][] matrix, int k) {
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a[0]));
        int n = matrix.length;
        for (int i = 0; i < n; i++) {
            pq.offer(new int[] {matrix[i][0], i, 0});
        }
        for (int i = 0; i < k - 1; i++) {
            int[] now = pq.poll();
            if (now[2] < n - 1) {
                pq.offer(new int[] {matrix[now[1]][now[2] + 1], now[1], now[2] + 1});
            }
        }
        return pq.poll()[0];
    }

    // -------有序矩阵中第K小的元素 << end --------

    // -------至少有K个重复字符的最长子串 start >>--------

    /**
     * 给你一个字符串 s 和一个整数 k ，请你找出 s 中的最长子串， 要求该子串中的每一字符出现次数都不少于 k 。返回这一子串的长度。
     *
     * 当确定了窗口内所包含的字符数量时，区间重新具有了二段性质。
     * 首先我们知道 【答案子串的左边界左侧的字符以及右边界右侧的字符一定不会出现在子串中，否则就不会是最优解】，但如果我们只从该性质出发的话，朴素解法
     * 应该是使用一个滑动窗口，不断的调整滑动窗口的左右边界，使其满足【左边界左侧的字符以及右边界右侧的字符一定不会出现在窗口中】，这实际上是双指针解法，
     * 但是如果不先敲定（枚举）出答案所包含的字符数量的话，这里的双指针是不具有单调性的。
     * 换句话说，只利用这一性质是没法完成逻辑的。
     * 这时候我们面临的问题是： 性质是正确的，但是还无法直接使用。
     * 因此我们需要先利用字符数量有限性（可枚举）作为切入点，是的【答案子串的左边界左侧的字符以及右边界右侧的字符一定不会出现在子串中】这一性质
     * 在双指针的实现下具有单调性
     *
     * 对应 leetcode中第 395 题。
     */
    public int longestSubstring(String s, int k) {
        int ans = 0, n = s.length();
        int[] cnt = new int[26];
        for (int p = 1; p <= 26; p++) {
            Arrays.fill(cnt, 0);
            // tot 代表[j, i] 区间所有的字符种类数量，sum代表满足【出现次数不少于k】的字符种类数量
            for (int i = 0, j = 0, tot = 0, sum = 0; i < n; i++) {
                int rightIndex = s.charAt(i) - 'a';
                cnt[rightIndex]++;
                // 如果添加到 cnt之后为1，说明字符总数+1
                if (cnt[rightIndex] == 1)
                    tot++;
                if (cnt[rightIndex] == k)
                    sum++;
                // 当区间所包含的字符种类数量tot超过了当前限定的数量p，那么我们要删除一些字母，即【左指针】右移
                while (tot > p) {
                    int leftIndex = s.charAt(j++) - 'a';
                    cnt[leftIndex]--;
                    if (cnt[leftIndex] == 0)
                        tot--;
                    if (cnt[leftIndex] == k - 1)
                        sum--;
                }
                // 当所有的字符都符合要求，更新答案
                if (tot == sum)
                    ans = Math.max(ans, i - j + 1);
            }
        }
        return ans;
    }

    // -------至少有K个重复字符的最长子串 << end --------

    ////// --------------helper class----------------

    class RandomNode {
        int val;
        RandomNode next;
        RandomNode random;

        RandomNode(int val) {
            this.val = val;
        }

        RandomNode(int val, RandomNode next) {
            this(val, next, null);
        }

        RandomNode(int val, RandomNode next, RandomNode random) {
            this.val = val;
            this.next = next;
            this.random = random;
        }

        @Override
        public String toString() {
            return "RandomNode{" +
                    "val=" + val +
                    '}';
        }
    }
}
