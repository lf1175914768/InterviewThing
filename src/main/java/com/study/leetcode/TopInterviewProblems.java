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

    // -------打家劫舍 start >>--------

    /**
     * 你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。
     * 给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。
     *
     * 对应 leetcode 中第 198 题。
     */
    public int rob(int[] nums) {
        if (nums.length == 0)
            return 0;
        if (nums.length == 1)
            return nums[0];
        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0], nums[1]);
        for (int i = 2; i < nums.length; i++) {
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i]);
        }
        return dp[nums.length - 1];
    }

    // -------打家劫舍 << end --------

    // -------缺失的第一个正数 start >>--------

    /**
     * 给你一个未排序的整数数组 nums ，请你找出其中没有出现的最小的正整数。
     * 请你实现时间复杂度为 O(n) 并且只使用常数级别额外空间的解决方案。
     *
     * 由于题目要求我们只能使用常数级别的空间，而要找的数据一定在[1, N + 1] 这个区间里，因此，我们可以就把原始的数组当做哈希表来使用。事实上，
     * 哈希表其实本身也是一个数组；
     * 我们要找的数就在[1, N + 1]里，最后N + 1这个元素我们不用找。因为在前面的 N 个元素都找不到的情况下，我们才返回 N + 1；
     * 那么，我们可以采取这样的思路：就把 1 这个数放到下标为0 的位置，2 这个数放到下标为1 的位置，按照这种思路遍历一遍数组。然后我们在遍历一次数组，
     * 第 1 个遇到的它的值不等于下标的那个数，就是我们要找的缺失的第一个正数。
     * 这个思想就相当于我们自己编写哈希函数，这个哈希函数的规则特别简单，那就是数值为 i 的数映射到下标为 i - 1 的位置。
     *
     * 对应 leetcode 中第 41 题。
     */
    public int firstMissingPositive(int[] nums) {
        int len = nums.length;
        for (int i = 0; i < len; i++) {
            while (nums[i] > 0 && nums[i] <= len && nums[nums[i] - 1] != nums[i]) {
                // 满足在指定范围内，并且没有放在正确的位置上，才交换
                // 例如： 数值3应该放在索引 2 的位置上
                int tmp = nums[nums[i] - 1];
                nums[nums[i] - 1] = nums[i];
                nums[i] = tmp;
            }
        }

        for (int i = 0; i < len; i++) {
            if (nums[i] != i + 1)
                return i + 1;
        }
        // 都正确的情况下 返回数组长度 + 1
        return len + 1;
    }

    // -------缺失的第一个正数 << end --------

    // -------通配符匹配 start >>--------

    /**
     * 给定一个字符串 (s) 和一个字符模式 (p) ，实现一个支持 '?' 和 '*' 的通配符匹配。
     *
     * '?' 可以匹配任何单个字符。
     * '*' 可以匹配任意字符串（包括空字符串）。
     * 两个字符串完全匹配才算匹配成功。
     * 说明:
     * s 可能为空，且只包含从 a-z 的小写字母。
     * p 可能为空，且只包含从 a-z 的小写字母，以及字符 ? 和 *。
     *
     * 解题思路：
     * 使用动态规划进行解答。其中 【小写字母】和【问号】的匹配是确定的，儿【星号】的匹配是不确定的，因此我们需要枚举所有的匹配情况。
     * 我们用 dp[i][j] 表示字符串 s 的前 i 个字符和模式 p 的前 j 个字符是否能匹配。在进行状态转移时，我们可以考虑模式 p 个第 j 个字符记为 pj， 与之对应的是
     * 字符串 s 中的第 i 个字符，记为 si：
     * 1. 如果 pj 是小写字母，那么 si 必须也为相同的小写字母，状态转移方程为：dp[i][j] = (si 与 pj 相同) && dp[i - 1][j - 1]
     * 2. 如果 pj 是问号，那么对 si 没有任何要求，状态转移方程为： dp[i][j] = dp[i - 1][j - 1]
     * 3. 如果 pj 是星号，那么同样对 si 没有任何要求，但是星号可以匹配零或任意多个小写字母，因此状态转移方程分为两种情况，即使用或不使用这个星号；
     *    dp[i][j] = dp[i][j - 1] || dp[i - 1][j]
     *   如果我们不适用这个星号，那么就会从 dp[i][j- 1] 转移而来；如果我们使用这个星号，那么就会从 dp[i - 1][j] 转移而来。
     *
     * 最终的状态方程如下：
     *            |  (si 与 pj 相同) && dp[i - 1][j - 1],  pj 是小写字母
     * dp[i][j] = |  dp[i - 1][j - 1],                    pj 是问号
     *            |  dp[i][j - 1] || dp[i - 1][j],        pj 是星号
     *
     * 确定边界条件：
     * 根据dp数组的定义，所有的 dp[0][j] 和 dp[i][0] 都是边界条件，因为它们涉及到空字符串或者空模式的情况。这是我们在状态方程中没有考虑到的：
     * dp[0][0] = True, 即当空字符串 s 和 模式 p 均为空时，匹配成功。
     * dp[i][0] = False, 即空模式无法匹配非空字符串。
     * dp[0][j] 需要分情况讨论：因为星号才能匹配空字符串，所以只有当模式 p 的前 j 个字符均为星号时， dp[0][j] 才为真。
     *
     * 我们可以发现，dp[i][0] 的值恒为假，dp[0][j]在 j 大于模式 p 的开头出现的星号字符个数之后，值也恒为假，而 dp[i][j] 的默认值（其他情况）也为假，
     * 因此在对动态规划的数组初始化时， 我们就可以将所有的状态初始化为 False，减少状态转移的代码编写难度。
     *
     * 最终的答案即为 dp[m][n], 其中 m 和 n 分别是字符串 s 和模式 p 的长度。需要注意的是，由于大部分语言中字符串的下标从 0 开始，
     * 因此 si 和 pj 分别对应这 s[i - 1] 和 p[j - 1]。
     *
     *
     * 对应 leetcode 中第 44 题。
     */
    public boolean isMatch(String s, String p) {
        int m = s.length();
        int n = p.length();
        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;
        for (int i = 1; i <= n; ++i) {
            if (p.charAt(i - 1) == '*') {
                dp[0][i] = true;
            } else {
                break;
            }
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (p.charAt(j - 1) == '*') {
                    dp[i][j] = dp[i][j - 1] || dp[i - 1][j];
                } else if (p.charAt(j - 1) == '?' || s.charAt(i - 1) == p.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                }
            }
        }
        return dp[m][n];
    }

    // -------通配符匹配 << end --------

    // -------组合 start >>--------

    /**
     * 给定两个整数 n 和 k，返回范围 [1, n] 中所有可能的 k 个数的组合。
     * 你可以按 任何顺序 返回答案。
     *
     * 这道题目属于 子集的 变体。相当于求 全部子集 中满足大小是 k 的子集的集合。
     * 只需要将base case 修改一下即可。
     *
     * @see HotProblems#subsets(int[])
     *
     * 对应 leetcode 中第 77 题。
     */
    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> res = new ArrayList<>();
        combineBackTrack(n, 1, k, new LinkedList<>(), res);
        return res;
    }

    private void combineBackTrack(int n, int start, int count, Deque<Integer> list, List<List<Integer>> res) {
        if (list.size() == count) {
            res.add(new ArrayList<>(list));
            return;
        }
        for (int i = start; i <= n; i++) {
            list.offerLast(i);
            combineBackTrack(n, i + 1, count, list, res);
            list.removeLast();
        }
    }

    // -------组合 << end --------

    // -------组合总和II start >>--------

    /**
     * 给定一个候选人编号的集合 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
     * candidates 中的每个数字在每个组合中只能使用 一次 。
     *
     * @see HotProblems#combinationSum_v2(int[], int)
     *
     * 对应 leetcode 中第 40 题。
     */
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        Arrays.sort(candidates);
        List<List<Integer>> res = new ArrayList<>();
        combinationSum2BackTrack(candidates, 0, target, new LinkedList<>(), res);
        return res;
    }

    private void combinationSum2BackTrack(int[] candidates, int start, int target, LinkedList<Integer> list, List<List<Integer>> res) {
        if (target == 0) {
            res.add(new ArrayList<>(list));
            return;
        }
        if (target < 0) {
            return;
        }
        for (int i = start; i < candidates.length; i++) {
            if (i > start && candidates[i] == candidates[i - 1])
                continue;
            list.offerLast(candidates[i]);
            combinationSum2BackTrack(candidates, i + 1, target - candidates[i], list, res);
            list.removeLast();
        }
    }

    // -------组合总和II << end --------

    // -------全排列II start >>--------

    /**
     * 给定一个可包含重复数字的序列 nums ，按任意顺序 返回所有不重复的全排列。
     *
     * 对应 leetcode 中第 47 题。
     */
    public List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        Arrays.sort(nums);
        boolean[] used = new boolean[nums.length];
        permuteUniqueBackTrack(nums, new LinkedList<>(), used, res);
        return res;
    }

    private void permuteUniqueBackTrack(int[] nums, Deque<Integer> list, boolean[] used, List<List<Integer>> res) {
        if (list.size() == nums.length) {
            res.add(new ArrayList<>(list));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (used[i])
                continue;
            /*
             * 这里进行剪枝
             *
             * 在标准的全排列算法中，比如 [1,2,2*] 和 [1,2*,2] 是不同的，但是在这里却是相同的排列，关键在于如何设计剪枝，将这种重复去除掉
             * 答案就是，保证相同元素在排列中的相对位置不变。
             * 就比如说 [1,2,2*] 这个例子，我只要保持 2 在 2* 前面不变。这样的话，就能满足条件
             * 标准全排列算法之所以出现重复，是因为把相同元素组成的排列序列视为不同的序列，但实际上他们应该是相同的；而如果固定相同元素形成的序列顺序，
             * 当然就避免了重复。
             * 当出现重复元素时，比如输入 [1,2,2*,2**], 2*只有在 2 已经被使用的情况下才会被选择， 2** 只有在 2* 已经被使用的情况下才会被选择。
             * 这样就保证了相同元素在排列中的相对位置保证固定。
             */
            if (i > 0 && nums[i] == nums[i - 1] && !used[i - 1])
                // 如果前面的相邻相等元素没有用过，则跳过
                continue;
            used[i] = true;
            list.offerLast(nums[i]);
            permuteUniqueBackTrack(nums, list, used, res);
            list.removeLast();
            used[i] = false;
        }
    }

    // -------全排列II << end --------

    // -------计数质数 start >>--------

    /**
     * 给定整数 n ，返回 所有小于非负整数 n 的质数的数量 。
     *
     * 对应 leetcode 中第 204 题。
     */
    public int countPrimes(int n) {
        boolean[] isPrime = new boolean[n];
        Arrays.fill(isPrime, true);
        for (int i = 2; i * i < n; i++) {
            if (isPrime[i]) {
                for (int j = i * i; j < n; j += i) {
                    isPrime[j] = false;
                }
            }
        }
        int count = 0;
        for (int i = 2; i < n; i++) {
            if (isPrime[i])
                count++;
        }
        return count;
    }

    // -------计数质数 << end --------

    // -------快乐数 start >>--------

    /**
     * 编写一个算法来判断一个数 n 是不是快乐数。
     * 「快乐数」 定义为：
     * 对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和。
     * 然后重复这个过程直到这个数变为 1，也可能是 无限循环 但始终变不到 1。
     * 如果这个过程 结果为 1，那么这个数就是快乐数。
     * 如果 n 是 快乐数 就返回 true ；不是，则返回 false 。
     *
     * 使用 ”快慢指针“ 的思想进行解答
     *
     * 对应 leetcode 中第 202 题。
     */
    public boolean isHappy(int n) {
        int slow = n, fast = n;
        do {
            slow = isHappyJudge(slow);
            fast = isHappyJudge(fast);
            fast = isHappyJudge(fast);
        } while (slow != fast);
        return slow == 1;
    }

    private int isHappyJudge(int n) {
        int sum = 0;
        while (n > 0) {
            int temp = n % 10;
            sum += temp * temp;
            n /= 10;
        }
        return sum;
    }

    // -------快乐数 << end --------

    // -------组合总和 start >>--------

    public List<List<Integer>> permute1(int[] nums) {
        return null;
    }

    // -------组合总和 << end --------

    ////// --------------helper class----------------

    static class RandomNode {
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
