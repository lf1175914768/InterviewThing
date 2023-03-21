package com.study.leetcode;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * <p>description: 动态规划相关的题目  </p>
 * <p>className:  DynamicProblems </p>
 * <p>create time:  2022/1/6 15:52 </p>
 *
 * @author feng.liu
 * @since 1.0
 **/
public class DynamicProblems {


    // -------最长递增子序列 start >>--------

    /**
     * 给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。
     * 子序列是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。
     * <p>
     * 解题思路：
     * 使用动态规划进行解题，设置 dp[i] 表示以 nums[i] 为最后一个元素的递增子序列，那么
     * dp[i] 怎么求呢？就是寻找 dp[0] 到 dp[i - 1] 中比 nums[i] 小并且 dp[j] 最大的一个，然后
     * dp[i] = dp[j] + 1
     * <p>
     * 对应 leetcode 中第 300题
     *
     * @param nums nums
     * @return max length of longest increasing subsequence
     */
    public int lengthOfLIS(int[] nums) {
        int[] dp = new int[nums.length];
        // base case: dp数组全部初始化为1
        Arrays.fill(dp, 1);
        int res = 1;
        for (int i = 1; i < nums.length; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j])
                    dp[i] = Math.max(dp[i], 1 + dp[j]);
            }
            res = Math.max(res, dp[i]);
        }
        return res;
    }

    /**
     * 新的状态定义：
     * 我们考虑维护一个列表 tails，其中每个元素 tails[k]的值代表长度为 k + 1 的子序列尾部的值
     * 转移方程：
     * 设 res 为tails 当前长度，代表直到当前的最长上升子序列长度。设 j 属于 [0,res), 考虑每轮遍历 nums[k] 时，通过二分法遍历 [0, res) 列表
     * 区间，找出 nums[k]的大小分界点，会出现两种情况：
     * 1、区间中存在 tails[i] > nums[k]: 将第一个满足 tails[i] > nums[k] 执行 tails[i] = nums[k]; 因为更小的 nums[k] 后更可能接一个
     * 比他大的数字。
     * 2、区间中不存在 tails[i] > nums[k]: 意味着 nums[k]可以接在前面所有长度的子序列之后，因此肯定是接到最长的后面（长度为res），新
     * 子序列长度为 res + 1.
     * 初始状态：
     * 令 tails 列表所有值 = 0
     * 返回值：
     * 返回res，即最长上升子序列的长度
     */
    public int lengthOfLIS_v2(int[] nums) {
        int[] tails = new int[nums.length];
        int res = 0;
        for (int num : nums) {
            int i = 0, j = res;
            while (i < j) {
                int m = (i + j) / 2;
                if (tails[m] < num)
                    i = m + 1;
                else
                    j = m;
            }
            tails[i] = num;
            if (res == j)
                res++;
        }
        return res;
    }

    // -------最长递增子序列 << end --------

    // -------目标和 start >>--------

    /**
     * 给你一个整数数组 nums 和一个整数 target 。
     * 向数组中的每个整数前添加 '+' 或 '-' ，然后串联起所有整数，可以构造一个 表达式 ：
     * 例如，nums = [2, 1] ，可以在 2 之前添加 '+' ，在 1 之前添加 '-' ，然后串联起来得到表达式 "+2-1" 。
     * 返回可以通过上述方法构造的、运算结果等于 target 的不同 表达式 的数目。
     * <p>
     * 解题思路：
     * 使用回溯的方法，进行解题
     * <p>
     * 对应 leetcode 中第 494 题
     */
    public int findTargetSumWays(int[] nums, int target) {
        if (nums.length == 0) return 0;
        AtomicInteger result = new AtomicInteger();
        findTargetSumWaysBackTrack(nums, 0, target, result);
        return result.get();
    }

    private void findTargetSumWaysBackTrack(int[] nums, int i, int rest, AtomicInteger result) {
        // base case
        if (i == nums.length) {
            if (rest == 0) {
                // 说明恰好凑出 target
                result.getAndIncrement();
            }
            return;
        }
        // 给 nums[i] 选择 “-” 号
        rest += nums[i];
        findTargetSumWaysBackTrack(nums, i + 1, rest, result);
        // 撤销选择
        rest -= nums[i];

        // 给 nums[i] 选择 “+” 号
        rest -= nums[i];
        findTargetSumWaysBackTrack(nums, i + 1, rest, result);
        // 撤销选择
//        rest += nums[i];
    }

    /**
     * 首先，我们把 nums 划分为两个子集 A 和 B，分别代表分配 + 的数和分配 - 的数，那么他们和 target 存在如下关系：
     * sum(A) - sum(B) = target
     * sum(A) = target + sum(B)
     * sum(A) + sum(A) = target + sum(B) + sum(A)
     * 2 * sum(A) = target + sum(nums)
     * <p>
     * 综上，可以推出 sum(A) = (target + sum(nums)) / 2, 也就是把原来的问题转化成： nums 存在几个子集A，使得 A 中元素的和为
     * (target + sum(nums)) / 2 ?
     * <p>
     * Note:
     * 上面的推导有缺陷，因为 背包问题需要 背包的容量是非负整数，但是 target + sum(nums) 有可能是负数，比如 nums = {100}, target = -200
     * 所以正式推导如下：
     * sum(A) - sum(B) = target
     * sum(A) = sum(nums) - sum(B)
     * sum(nums) - 2 * sum(B) = target
     * sum(B) = (sum(nums) - target) / 2
     * 由于数组 nums 中的元素都是非负整数, sum(B) 必须是非负偶数，若不成立，可直接返回 0，这样的话，可以将上面的问题解决掉
     * <p>
     * 好的，变成标准的背包问题：
     * 有一个背包，容量为 sum，现在给你N个物品，第 i个物品的重量为 nums[i - 1] (1 <= i <= N),每个物品只有一个，请问你有几种不同的方法能够恰好装满这个背包？
     * <p>
     * 第一步要明确两点：【状态】和【选择】。
     * 对于背包问题，这个都是一样的，状态就是【背包的容量】和【可选择的物品】，选择就是【装进背包】和【不装进背包】。
     * 第二步要明确 dp 数组的定义：
     * dp[i][j] = x 表示，若只在前 i 个物品中选择， 若当前背包的容量为 j，则最多有x中方法可以恰好装满背包。
     * 根据这个定义，显然 dp[0][..] = 0,因为没有物品的话，根本没办法装背包； dp[..][0] = 1, 因为如果背包的最大载重为 0，【什么都不装】就是唯一的一种装法。
     * 我们所求的答案就是 dp[N][sum]， 即使用所有 N个物品，有几种方法可以装满容量为 sum 的背包。
     * 第三步，根据【选择】，思考状态转移的逻辑。
     * 如果不把 nums[i] 算入子集，或者说你不把这第 i 个物品装入背包，那么恰好装满背包的方法就取决于上一个状态 dp[i - 1][j],继承之前的结果。
     * 如果把 nums[i] 算入子集，或者说你把这第 i 个物品装入背包，那么只要看前 i-1 个物品有几种方法可以装满 j - nums[i - 1] 的重量就行了，所以取决于 dp[i-1][j-nums[i-1]]
     * ps: 注意我们说的 i 是从1开始的，而数组 nums 的索引是从 0 开始算的，所以 nums[i - 1]代表的是第 i 个物品的重量，j - nums[i - 1]就是背包装入物品 i之后还剩下的重量。
     * 由于 dp[i][j] 为装满背包的总方法数，所以应该以上两种选择的结果求和，得到状态转移方程
     * dp[i][j] = dp[i-1][j] + dp[i-1][j-nums[i-1]]
     */
    public int findTargetSumWays_v2(int[] nums, int target) {
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }
        // 这两种情况，不可能存在合法的子集划分
        if (sum < target || (target + sum) % 2 == 1)
            return 0;
        return findTargetSumWaysSubSet(nums, (sum - target) / 2);
    }

    private int findTargetSumWaysSubSet(int[] nums, int sum) {
        int n = nums.length;
        int[][] dp = new int[n + 1][sum + 1];
        // base case
        for (int i = 0; i <= n; i++) {
            dp[i][0] = 1;
        }
        for (int i = 1; i <= n; i++) {
            for (int j = 0; j <= sum; j++) {
                if (j >= nums[i - 1]) {
                    // 两种选择的结果之和
                    dp[i][j] = dp[i - 1][j] + dp[i - 1][j - nums[i - 1]];
                } else {
                    // 背包的空间不足，只能选择不装物品 i
                    dp[i][j] = dp[i - 1][j];
                }
            }
        }
        return dp[n][sum];
    }

    /**
     * 这个是上面 第二种方法的 空间压缩版
     */
    public int findTargetSumWays_v3(int[] nums, int target) {
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }
        if (sum < target || (target + sum) % 2 == 1)
            return 0;

        int n = nums.length, total = (sum - target) / 2;
        int[] dp = new int[total + 1];
        // base case
        dp[0] = 1;
        for (int i = 1; i <= n; i++) {
            // j 要从后往前遍历
            for (int j = total; j >= 0; j--) {
                if (j - nums[i - 1] >= 0) {
                    dp[j] = dp[j] + dp[j - nums[i - 1]];
                }
            }
        }
        return dp[total];
    }

    // -------目标和 << end --------

    // -------下降路径最小和 start >>--------

    /**
     * 下降路径最小和
     * <p>
     * 对应 leetcode 中第931题
     */
    public int minFallingPathSum(int[][] matrix) {
        int[] dp = new int[matrix[0].length];
        for (int i = 0; i < dp.length; i++) {
            dp[i] = matrix[0][i];
        }
        int row = 0, res = Integer.MAX_VALUE;
        while (++row < matrix.length) {
            int pre = dp[0];
            for (int i = 0; i < dp.length; i++) {
                int minValue = dp[i];
                if (i > 0) {
                    minValue = Math.min(minValue, pre);
                }
                if (i < dp.length - 1) {
                    minValue = Math.min(minValue, dp[i + 1]);
                }
                pre = dp[i];
                dp[i] = matrix[row][i] + minValue;
            }
        }
        for (int i = 0; i < dp.length; i++) {
            res = Math.min(res, dp[i]);
        }
        return res;
    }

    // -------下降路径最小和 << end --------

    // -------编辑距离 start >>--------

    /**
     * 给你两个单词 word1 和 word2，请你计算出将 word1 转换成 word2 所使用的最少操作数。
     * 你可以对一个单词进行如下三种操作：
     * 1、插入一个字符
     * 2、删除一个字符
     * 3、替换一个字符
     * <p>
     * 对应 leetcode 中第 72 题
     */
    public int minDistance(String word1, String word2) {
        return minDistanceTraverse(word1, word1.length() - 1, word2, word2.length() - 1);
    }

    /**
     * 定义函数的返回值是 word1[0..word1end] 和 word2[0..word2end]的最小编辑距离
     */
    private int minDistanceTraverse(String word1, int word1end, String word2, int word2end) {
        // base case
        if (word1end == -1)
            return word2end + 1;
        if (word2end == -1)
            return word1end + 1;
        // 本来就相等，不需要任何操作
        // word1[0..word1end] 和 word2[0..word2end] 的最小编辑距离就等于
        // word1[0..word1end - 1] 和 word2[0..word2end - 1]
        if (word1.charAt(word1end) == word2.charAt(word2end))
            return minDistanceTraverse(word1, word1end - 1, word2, word2end - 1);
        // 我直接在 word1[word1end] 插入一个和 word2[word2end] 一样的字符，那么 word2[word2end] 就被匹配了
        // 前移 word2end ,继续跟 word1end 对比
        int insert = minDistanceTraverse(word1, word1end, word2, word2end - 1);
        // 直接把 word1[word1end] 这个字符删掉，前移 word1end， 继续跟 word2end 对比
        int delete = minDistanceTraverse(word1, word1end - 1, word2, word2end);
        // 直接把 word1[word1end] 替换成 word2[word2end], 这样的话，他俩就匹配了，同时前移 word1end, word2end 继续对比
        int replace = minDistanceTraverse(word1, word1end - 1, word2, word2end - 1);
        return Math.min(insert, Math.min(delete, replace)) + 1;
    }

    /**
     * 动态规划解法：
     * dp数组的定义如下：
     * dp[i-1][j-1] 存储 s1[0..i] 和 s2[0..j] 的最小编辑距离
     * 但是 dp 数组的base case 是 i，j 等于 -1，而数组索引至少是0，所以dp数组会偏移一位。
     */
    public int minDistance_v2(String word1, String word2) {
        int m = word1.length(), n = word2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++)
            dp[i][0] = i;
        for (int j = 1; j <= n; j++)
            dp[0][j] = j;
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    int min = Math.min(dp[i][j - 1], dp[i - 1][j]);
                    dp[i][j] = Math.min(min, dp[i - 1][j - 1]) + 1;
                }
            }
        }
        return dp[m][n];
    }

    /**
     * 上面的解法的 dp 数组压缩版
     */
    public int minDistance_v3(String word1, String word2) {
        int m = word1.length(), n = word2.length();
        int len = Math.min(m, n), maxCount = Math.max(m, n);
        boolean mIsShort = len == m;
        String sShort = mIsShort ? word1 : word2, sLong = mIsShort ? word2 : word1;
        int[] dp = new int[len + 1];
        for (int i = 0; i <= len; i++) {
            dp[i] = i;
        }
        for (int i = 1; i <= maxCount; i++) {
            dp[0] = i;
            int pre = i - 1;
            for (int j = 1; j <= len; j++) {
                int preOn = dp[j];
                if (sLong.charAt(i - 1) == sShort.charAt(j - 1)) {
                    dp[j] = pre;
                } else {
                    dp[j] = Math.min(dp[j], Math.min(dp[j - 1], pre)) + 1;
                }
                pre = preOn;
            }
        }
        return dp[len];
    }

    // -------编辑距离 << end --------

    // -------俄罗斯套娃信封问题 start >>--------

    /**
     * 给你一个二维整数数组 envelopes ，其中 envelopes[i] = [wi, hi] ，表示第 i 个信封的宽度和高度。
     * 当另一个信封的宽度和高度都比这个信封大的时候，这个信封就可以放进另一个信封里，如同俄罗斯套娃一样。
     * 请计算 最多能有多少个 信封能组成一组“俄罗斯套娃”信封（即可以把一个信封放到另一个信封里面）。
     * 注意：不允许旋转信封。
     * <p>
     * 这道题的解法是比较巧妙的:
     * 先对宽度 w 进行升序排列，如果遇到 w 相同的情况，则按照高度 h 降序排列。之后把所有的 h 作为一个数组，
     * 在这个数组上计算 LIS 的长度就是答案。
     * 这个解法的关键在于，对于宽度 w 相同的数对，要对其高度 h 进行降序排列。因为两个宽度相同的信封不能相互包含的。
     * 逆序排列保证在 w 相同的数对中最多只选取一个。
     * <p>
     * 对应 leetcode 中第 354 题
     */
    public int maxEnvelopes(int[][] envelopes) {
        int n = envelopes.length;
        // 按宽度升序排列，如果宽度一样，则按高度降序排列
        Arrays.sort(envelopes, (a, b) -> a[0] == b[0] ?
                b[1] - a[1] : a[0] - b[0]);
        int[] height = new int[n];
        for (int i = 0; i < n; i++) {
            height[i] = envelopes[i][1];
        }

        int result = 0;
        for (int num : height) {
            int i = 0, j = result;
            while (i < j) {
                int mid = i + (j - i) / 2;
                if (height[mid] < num) {
                    i = mid + 1;
                } else {
                    j = mid;
                }
            }
            height[i] = num;
            if (j == result)
                result++;
        }
        return result;
    }

    // -------俄罗斯套娃信封问题 << end --------

    // -------最大子数组和 start >>--------

    /**
     * 给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
     * 子数组 是数组中的一个连续部分。
     * <p>
     * 解题思路：
     * 首先想到的动态规划，一般来说是这样定义 `dp` 数组的：
     * nums[0..i] 中的 【最大的子数组和】 为 dp[i]。
     * 如果这样定义的话，整个nums 数组的 【最大子数组和】就是 dp[n-1]。如何找状态转移方程呢？按照数学归纳发法，
     * 假设我们知道了 dp[i - 1], 如何推导出 dp[i] 呢？
     * 实际上是不行的， 因为子数组一定是连续的，按照我们当前`dp`数组的定义，并不能保证 nums[0..i]中的最大子数组
     * 与 nums[i+1] 是相邻的，也就没办法从 dp[i] 推导出 dp[i+1].
     * <p>
     * 所以我们重新定义一下 dp 数组的含义：
     * 以 nums[i] 为结尾的 【最大子数组和】 为 dp[i]。
     * 这种定义下，想得到整个 nums 数组的 【最大子数组和】，不能直接返回 dp[n+1], 而需要遍历整个 dp 数组。
     * 依然使用数学归纳法来找 状态转移关系： 假设我们已经算出了 dp[i-1], 如何推导出 dp[i] 呢？
     * 可以做到， dp[i] 有两种选择，要么与前面的相邻子数组连接，形成一个和更大的子数组；要么不与前面的子数组连接，自成一派，
     * 自己作为一个子数组。
     * <p>
     * 对应 leetcode 中第 53 题
     */
    public int maxSubArray(int[] nums) {
        int n = nums.length;
        if (n == 0) return 0;
        int[] dp = new int[n];
        // base case
        // 第一个元素前面没有子数组
        dp[0] = nums[0];
        int result = dp[0];
        for (int i = 1; i < n; i++) {
            dp[i] = Math.max(nums[i], nums[i] + dp[i - 1]);
            result = Math.max(result, dp[i]);
        }
        return result;
    }

    /**
     * 上面解题方法的 dp 数组压缩版
     */
    public int maxSubArray_v2(int[] nums) {
        int n = nums.length;
        if (n == 0) return 0;
        int result = nums[0], dp = nums[0];
        for (int i = 1; i < n; i++) {
            dp = Math.max(nums[i], nums[i] + dp);
            result = Math.max(result, dp);
        }
        return result;
    }

    // -------最大子数组和 << end --------

    // -------最长公共子序列 start >>--------

    /**
     * 给定两个字符串 text1 和 text2，返回这两个字符串的最长 公共子序列 的长度。如果不存在 公共子序列 ，返回 0 。
     * 一个字符串的 子序列 是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。
     * 例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。
     * 两个字符串的 公共子序列 是这两个字符串所共同拥有的子序列。
     * <p>
     * 解题思路：
     * 我们定义一个 dp 函数： 计算 s1[i..] 和 s2[j..] 的最长公共子序列长度
     * int dp(String s1, int i, String s2, int j)
     * 那我们想要的答案就是 dp(s1, 0, s2, 0), 且base case就是 i == len(s1) 或 j == len(s2)，返回值是0
     * 接下来，咱不要看 s1 和 s2两个字符串，而是要具体到每一个字符，思考每个字符要做什么。
     * 如果 s1[i] == s2[j],说明这个字符一定在 lcs 中，这样，就找到了一个 lcs 中的字符。
     * 如果 s1[i] != s2[j],那便意味着，s1[i] 和 s2[j] 中至少有一个字符不在 lcs 中，所以有三种情况：
     * 1、s1[i] 不在 lcs 中
     * 2、s2[j] 不在 lcs 中
     * 3、都不在 lcs 中
     * 另外还有一个小的优化，情况三其实可以直接忽略，因为我们在求最大值嘛，情况三在计算 s1[i+1..] 和 s2[j+1..]
     * 的 lcs 长度，这个长度肯定是 情况二 和 情况一的。说白了，情况三被情况一和情况二包含了。
     * <p>
     * 对应 leetcode 中第 1143 题
     */
    public int longestCommonSubsequence(String text1, String text2) {
        int m = text1.length(), n = text2.length();
        int[][] memo = new int[m][n];
        // 备忘录值 为 -1，代表未曾计算
        for (int[] row : memo) {
            Arrays.fill(row, -1);
        }
        // 计算 s1[0..] 和 s2[0..] 的 lcs长度
        return longestCommonSubsequenceDP(text1, 0, text2, 0, memo);
    }

    /**
     * 定义： 计算 s1[i..] 和 s2[j..] 的最长公共子序列
     */
    private int longestCommonSubsequenceDP(String s1, int i, String s2, int j, int[][] memo) {
        if (i == s1.length() || j == s2.length())
            return 0;
        // 如果之前计算过，则直接返回备忘录中的答案
        if (memo[i][j] != -1) {
            return memo[i][j];
        }
        if (s1.charAt(i) == s2.charAt(j)) {
            // s1[i] 和 s2[j] 必然在 lcs 中
            memo[i][j] = 1 + longestCommonSubsequenceDP(s1, i + 1, s2, j + 1, memo);
        } else {
            // s1[i] 和 s2[j] 至少有一个不在 lcs 中
            memo[i][j] = Math.max(
                    longestCommonSubsequenceDP(s1, i, s2, j + 1, memo),
                    longestCommonSubsequenceDP(s1, i + 1, s2, j, memo)
            );
        }
        return memo[i][j];
    }

    /**
     * 使用自底向上的 迭代的动态规划思路。
     * 定义dp数组：
     * dp[i][j] 表示 s1[0..i-1] 和 s2[0..j-1] 的最长公共子序列
     */
    public int longestCommonSubsequence_v2(String text1, String text2) {
        int m = text1.length(), n = text2.length();
        int[][] dp = new int[m+1][n+1];
        // base case: dp[0][..] = dp[..][0] = 0

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
                    dp[i][j] = 1 + dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.max(dp[i][j - 1], dp[i - 1][j]);
                }
            }
        }
        return dp[m][n];
    }

    /**
     * 上面解法的 dp 压缩版本
     */
    public int longestCommonSubsequence_v3(String text1, String text2) {
        int m = text1.length(), n = text2.length();
        int shortCount = Math.min(m, n), longCount = Math.max(m, n);
        boolean mIsShort = m == shortCount;
        String sShort = mIsShort ? text1 : text2, sLong = mIsShort ? text2 : text1;
        int[] dp = new int[shortCount + 1];
        for (int i = 1; i <= longCount; i++) {
            int pre = dp[0];
            for (int j = 1; j <= shortCount; j++) {
                int preOn = dp[j];
                if (sLong.charAt(i - 1) == sShort.charAt(j - 1)) {
                    dp[j] = 1 + pre;
                } else {
                    dp[j] = Math.max(dp[j - 1], dp[j]);
                }
                pre = preOn;
            }
        }
        return dp[shortCount];
    }

    // -------最长公共子序列 << end --------

    // -------得到子序列的最少操作次数 start >>--------

    /**
     * 给你一个数组 target ，包含若干 互不相同 的整数，以及另一个整数数组 arr ，arr 可能 包含重复元素。
     * 每一次操作中，你可以在 arr 的任意位置插入任一整数。比方说，如果 arr = [1,4,1,2] ，那么你可以在中间添加 3 得到 [1,4,3,1,2] 。
     * 你可以在数组最开始或最后面添加整数。
     * 请你返回 最少 操作次数，使得 target 成为 arr 的一个子序列。
     * 一个数组的 子序列 指的是删除原数组的某些元素（可能一个元素都不删除），同时不改变其余元素的相对顺序得到的数组。
     * 比方说，[2,7,4] 是 [4,2,3,7,2,1,4] 的子序列（加粗元素），但 [2,4,2] 不是子序列。
     * <p/>
     * 为了方便，我们令 target 长度为 n，arr长度为 m，target 和 arr的最长公共子序列长度为 max，不难发现最终答案为 n - max。
     * 因此从题面来说，这是一道最长公共子序列问题 （LCS）.
     * 但朴素求解 LCS问题复杂度为 O(m*n)。使用状态定义 【f[i][j]为考虑 a 数组的前 i 个元素和 b 数组的前 j 个元素的最长公共子序列长度为多少】进行求解。
     * <p/>
     * 当 LCS 问题添加某些条件限制之后，会存在一些很有趣的性质：
     * 其中一个经典的性质就是：当其中一个数组元素各不相同时，最长公共子序列问题 (LCS) 可以转化为最长上升子序列 (LIS) 进行求解。同时最长上升子序列
     * 问题 (LIS) 存在使用 【维护单调序列 + 二分】的贪心解法，复杂度为 O(nlogn)。
     * 因此本题可以通过 [LCS]问题 -> 利用 target 数组元素各不相同时，转化为 LIS 问题 -> 使用 LIS 的贪心算法，做到 O(nlogn) 的复杂度。
     * <p/>
     * <p>
     *  证明：
     *  1、为何其中一个数组元素各不相同时，LCS问题可以转化为 LIS 问题？
     *  Answer：本质是利用【当其中一个数组元素各不相同时，这时候每一个“公共子序列”都对应一个不重复元素数组的下标数组“上升子序列”，反之亦然】。
     * </p>
     *
     * 对应 leetcode 中第 1713 题。
     */
    public int minOperations(int[] target, int[] arr) {
        int n = target.length;
        Map<Integer, Integer> pos = new HashMap<>();
        for (int i = 0; i < n; i++) {
            pos.put(target[i], i);
        }
        List<Integer> resList = new ArrayList<>();
        for (int val : arr) {
            if (pos.containsKey(val)) {
                int idx = pos.get(val);
                int it = minOperationBinarySearch(resList, idx);
                if (it != resList.size()) {
                    resList.set(it, idx);
                } else {
                    resList.add(idx);
                }
            }
        }
        return n - resList.size();
    }

    private int minOperationBinarySearch(List<Integer> resList, int target) {
        int size = resList.size();
        if (size == 0 || resList.get(size - 1) < target) {
            return size;
        }
        int low = 0, high = size - 1;
        while (low < high) {
            int middle = low + ((high - low) >>> 1);
            if (resList.get(middle) < target) {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        return low;
    }

    // -------得到子序列的最少操作次数 << end --------

    // -------两个字符串的删除操作 start >>--------

    /**
     * 给定两个字符串 s1 和 s2，分别删除若干字符串之后使得两个字符串相同，则剩下的字符为两个字符串的公共子序列。为了使得删除操作的次数最少，
     * 剩下的字符尽可能多。当剩下的字符为两个字符串的 最长公共子序列 时，删除操作的次数最少。因此，可以计算两个字符串的最长公共子序列的长度，
     * 然后分别计算两个字符串的长度和最长公共子序列的长度之差，即为两个字符分别需要删除的次数，然后相加即可。
     *
     * 对应  leetcode 中第 583 题
     */
    public int minDistanceOfDeletion(String word1, String word2) {
        int m = word1.length(), n = word2.length();
        int[][] dp = new int[m+1][n+1];

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = 1 + dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        int commonCount = dp[m][n];
        return m - commonCount + n - commonCount;
    }

    /**
     * 可以直接计算最少删除次数， 其中 dp 数组的定义如下：
     * dp[i][j] 表示使 s1[0..i] 和 s2[0..j] 相同的最少删除次数
     * 动态规划的边界情况如下：
     * 当 i == 0 时，s1[0..i] 为空，空字符串和任何字符串要变成相同，只有将另一个非空字符串全部删除掉，因此对任意 0 <= j <= n, 有 dp[0][j] = j;
     * 当 j == 0 时，s2[0..j] 为空，同理可得, 对任意 0 <= i <= m, 有 dp[i][0] = i;
     *
     * 当 i > 0 且 j > 0时，考虑 dp[i][j] 的计算：
     *  1、当 s1[i - 1] == s2[j - 1] 时，将这两个相同的字符成为公共字符，考虑使 s1[0..i-1] 和 s2[0..j-1] 相同的最少删除次数，增加一个公共字符之后，
     *     最少删除的次数不变， 因此 dp[i][j] = dp[i-1][j-1].
     *  2、当 s1[i - 1] != s2[j - 1] 时，有两种情况：
     *     2.1、使得 s1[0..i-1] 和 s2[0..j] 相同的组少删除次数，加上删除 s1[i-1] 的 1 次操作。
     *     2.2、使得 s1[0..i] 和 s2[0..j-1] 相同的组少删除次数，加上删除 s2[j-1] 的 1 次操作。
     *  要得到使 s1[0..i] 和 s2[0..j] 相同的最少删除次数，应取两项中的较小一项。
     */
    public int minDistanceOfDeletion_v2(String word1, String word2) {
        int m = word1.length(), n = word2.length();
        int[][] dp = new int[m+1][n+1];
        for (int i = 1; i <= m; i++) {
            dp[i][0] = i;
        }
        for (int i = 1; i <= n; i++) {
            dp[0][i] = i;
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = 1 + Math.min(dp[i][j - 1], dp[i - 1][j]);
                }
            }
        }
        return dp[m][n];
    }

    // -------两个字符串的删除操作 << end --------

    // -------零钱兑换II start >>--------

    /**
     * 给你一个整数数组 coins 表示不同面额的硬币，另给一个整数 amount 表示总金额。
     * 请你计算并返回可以凑成总金额的硬币组合数。如果任何硬币组合都无法凑出总金额，返回 0 。
     * 假设每一种面额的硬币有无限个。
     * 题目数据保证结果符合 32 位带符号整数
     *
     * 解题思路：背包问题：
     * 第一步明确两点： 【状态】 和 【选择】
     * 状态有两个，就是【背包的容量】和【可选择的物品】，选择就是【装进背包】和【不装进背包】
     * 第二步明确 dp 数组的定义：
     * dp[i][j]表示若只是用前 i 个物品（可以重复使用），当背包容量为 j 时，有 dp[i][j] 中方法可以装满背包
     * 或者说： 若只使用 coins 中的前 i 个硬币的面值，若想凑出金额 j，有 dp[i][j] 中凑法。
     * 明显 base case 是 dp[0][..] = 0, dp[..][0] = 1,因为如果不适用任何硬币面值，就无法凑出任何金额；
     * 如果凑出的金额是 0，那么‘无为而治’就是唯一的解法。 我们最终要得到的答案就是 dp[N][amount], 其中 N 为coins 数组的大小。
     * 第三步：根据【选择】，思考状态转移方程的逻辑。
     * 如果你不把这第 i 个物品装入背包，也就是说你不使用 coins[i]这个面值的硬币，那么凑出面额 j 的方法数 dp[i][j]
     * 应该等于 dp[i-1][j], 继承之前的结果。
     * 如果你把这第 i 个物品装入背包，也就是说你使用 coins[i] 这个面值的硬币，那么 dp[i][j] 应该等于 dp[i][j-coins[i-1]].
     * 首先 i 是从 1 开始的，所以coins的索引是 i - 1 时表示第 i个硬币的面值。
     * 综上就是两种选择，而我们想要求的 dp[i][j] 是【共有多少种解法】，所以 dp[i][j] 的值是以上两种选择的结果之和。
     *
     * 对应 leetcode 中第 518 题
     */
    public int change(int amount, int[] coins) {
        int n = coins.length;
        int[][] dp = new int[n+1][amount+1];
        for (int i = 1; i <= n; i++) {
            dp[i][0] = 1;
        }
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= amount; j++) {
                if (j >= coins[i-1]) {
                    dp[i][j] = dp[i-1][j] + dp[i][j - coins[i-1]];
                } else {
                    dp[i][j] = dp[i-1][j];
                }
            }
        }
        return dp[n][amount];
    }

    /**
     * 上面解法的 dp 数组压缩版
     */
    public int change_v2(int amount, int[] coins) {
        int n = coins.length;
        int[] dp = new int[amount + 1];
        dp[0] = 1;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= amount; j++) {
                if (j >= coins[i - 1]) {
                    dp[j] += dp[j - coins[i - 1]];
                }
            }
        }
        return dp[amount];
    }

    // -------零钱兑换II << end --------

    // -------打家劫舍II start >>--------

    /**
     * 你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋都 围成一圈 ，这意味着第一个房屋和最后一个房屋是紧挨着的。同时，相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警 。
     * 给定一个代表每个房屋存放金额的非负整数数组，计算你 在不触动警报装置的情况下 ，今晚能够偷窃到的最高金额。
     *
     * 解题思路：
     * 相比于 【打家劫舍】198 题，
     * 这个问题只是边界这个多了一个限制，其他的条件没有变，所以还是地地道道的动态规划问题，只不过需要考虑一下边界条件。
     * 我们可以打破首尾连续的情况即可，害怕第一户和最后一户连续，那么拆开讨论就行了，进行两次动态规划：
     * 第一户人家如果不偷，那么最后一户人家是可偷可不偷的，那么可以初始化第一户不偷 dp 跑到最后一家获得一个不偷的最大值。
     * 第一户人家如果偷，那么最后一户人家一定不能偷（否则连续），但是倒数第二家无所谓，可偷可不偷，那么可以初始第一家偷 dp 跑到倒数第二家
     * 获得一个第一户偷的最大值。
     * 然后就是取两种情况的最大值就是获得的结果了。
     *
     * 对应 leetcode 中第 213 题。
     */
    public int rob2(int[] nums) {
        int[] dp = new int[nums.length + 1];
        dp[0] = 0;
        dp[1] = nums[0]; // 偷第一户
        for (int i = 2; i < nums.length; i++) {
            dp[i] = Math.max(nums[i - 1] + dp[i - 2], dp[i - 1]);
        }
        int val = dp[nums.length - 1];  // 前 len - 1户的dp值
        dp[1] = 0;   // 第一户不偷
        for (int i = 2; i < nums.length + 1; i++) {
            dp[i] = Math.max(nums[i - 1] + dp[i - 2], dp[i - 1]);
        }
        val = Math.max(val, dp[nums.length]);
        return val;
    }

    // -------打家劫舍II << end --------

    // -------最大正方形 start >>--------

    /**
     * 在一个由 '0' 和 '1' 组成的二维矩阵内，找到只包含 '1' 的最大正方形，并返回其面积。
     *
     * 使用动态规划得方法进行解答：
     * 定义 dp[i][j] 是以 matrix[i - 1][j - 1] 为右下角得正方形得最大边长。
     * 则可以根据画图帮助理解，有如下性质：
     * 若某格子值为 1，则以此为右下角得正方形得，最大边长为：上面得正方形、左面得正方形或左上的正方形中，最小的那个，在加上此格。
     * 则有：
     * <pre>
     *   if (grid[i - 1][j - 1] == '1') {
     *       dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1;
     *   }
     * </pre>
     *
     * 对应 leetcode 中第 221 题。
     */
    public int maximalSquare(char[][] matrix) {
        if (null == matrix || matrix.length <= 0) return 0;
        int[][] dp = new int[matrix.length + 1][matrix[0].length + 1];
        int maxLength = 0;
        for (int i = 1; i <= matrix.length; i++) {
            for (int j = 1; j <= matrix[0].length; j++) {
                if (matrix[i - 1][j - 1] == '1') {
                    dp[i][j] = Math.min(Math.min(dp[i - 1][j - 1], dp[i - 1][j]), dp[i][j - 1]) + 1;
                    maxLength = Math.max(maxLength, dp[i][j]);
                }
            }
        }
        return maxLength * maxLength;
    }

    // -------最大正方形 << end --------

    // -------不同的子序列II start >>--------

    /**
     * 给定一个字符串 s，计算 s 的 不同非空子序列 的个数。因为结果可能很大，所以返回答案需要对 10^9 + 7 取余 。
     * 字符串的 子序列 是经由原字符串删除一些（也可能不删除）字符但不改变剩余字符相对位置的一个新字符串。
     *
     * 我们考虑使用动态规划的方法进行解答。
     * 我们可以设 dp[i] 为前 i 个字符可以组成的不同的子序列，则有 dp[i] = dp[i - 1] + newCount - repeatCount
     * 其中 newCount 为加上 s[i] 后新增的子序列个数， repeatCount 为重复的子序列个数。
     *
     * newCount 的值比较好算，就是 dp[i - 1].
     * 举例：设有字符串 abcb
     * 状态：      子序列：
     * 没有遍历时：  空串 “”
     * 遍历到 a 时： "", a
     * 遍历到 b 时： "", a, b, ab   -> 后两个子序列由上一次的结果集 + 当前字符构成
     * 遍历到 c 时： "", a, b, ab, c, ac, bc, abc
     * 遍历到 b 时： "", a, b, ab, c, ac, bc, abc, b, ab, bb, abc, cb, acb, bcb, abcb
     *
     * 然后我们计算 repeatCount，我们观察遍历到的第二个字符 b，出现重复的序列为 b 和 ab，而这两个序列正好是上一次添加 b 的时候新增的两个序列。
     * 于是我们可以使用数组 preCount 来记录上一次该字符串新增的个数，该个数就是 repeatCount。
     *
     * 最后，我们将空串减去即可。
     *
     * 对应 leetcode 中第 940 题。
     */
    public int distinctSubSeqII(String s) {
        int maxMod = (int) (1e9 + 7);
        int[] preCount = new int[26];
        int curAns = 1;   // 没有遍历时，空串
        for (int i = 0; i < s.length(); i++) {
            // 新增的个数
            int newCount = curAns;
            char c = s.charAt(i);
            curAns = ((curAns + newCount) % maxMod - preCount[c - 'a'] % maxMod + maxMod) % maxMod;
            // 记录当前字符的新增值
            preCount[c - 'a'] = newCount;
        }

        // 减去空串
        return curAns - 1;
    }

    // -------不同的子序列II << end --------

    // -------组合总和 start >>--------

    public List<List<Integer>> permute1(int[] nums) {
        return null;
    }

    // -------组合总和 << end --------

}
