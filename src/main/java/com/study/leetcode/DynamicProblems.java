package com.study.leetcode;

import java.util.Arrays;
import java.util.List;
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
     *
     * 解题思路：
     * 使用动态规划进行解题，设置 dp[i] 表示以 nums[i] 为最后一个元素的递增子序列，那么
     * dp[i] 怎么求呢？就是寻找 dp[0] 到 dp[i - 1] 中比 nums[i] 小并且 dp[j] 最大的一个，然后
     * dp[i] = dp[j] + 1
     *
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
     *  我们考虑维护一个列表 tails，其中每个元素 tails[k]的值代表长度为 k + 1 的子序列尾部的值
     * 转移方程：
     *  设 res 为tails 当前长度，代表直到当前的最长上升子序列长度。设 j 属于 [0,res), 考虑每轮遍历 nums[k] 时，通过二分法遍历 [0, res) 列表
     *  区间，找出 nums[k]的大小分界点，会出现两种情况：
     *  1、区间中存在 tails[i] > nums[k]: 将第一个满足 tails[i] > nums[k] 执行 tails[i] = nums[k]; 因为更小的 nums[k] 后更可能接一个
     *  比他大的数字。
     *  2、区间中不存在 tails[i] > nums[k]: 意味着 nums[k]可以接在前面所有长度的子序列之后，因此肯定是接到最长的后面（长度为res），新
     *  子序列长度为 res + 1.
     * 初始状态：
     *  令 tails 列表所有值 = 0
     * 返回值：
     *  返回res，即最长上升子序列的长度
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
     *
     * 解题思路：
     * 使用回溯的方法，进行解题
     *
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
     *
     * 综上，可以推出 sum(A) = (target + sum(nums)) / 2, 也就是把原来的问题转化成： nums 存在几个子集A，使得 A 中元素的和为
     * (target + sum(nums)) / 2 ?
     *
     * Note:
     * 上面的推导有缺陷，因为 背包问题需要 背包的容量是非负整数，但是 target + sum(nums) 有可能是负数，比如 nums = {100}, target = -200
     * 所以正式推导如下：
     * sum(A) - sum(B) = target
     * sum(A) = sum(nums) - sum(B)
     * sum(nums) - 2 * sum(B) = target
     * sum(B) = (sum(nums) - target) / 2
     * 由于数组 nums 中的元素都是非负整数, sum(B) 必须是非负偶数，若不成立，可直接返回 0，这样的话，可以将上面的问题解决掉
     *
     * 好的，变成标准的背包问题：
     * 有一个背包，容量为 sum，现在给你N个物品，第 i个物品的重量为 nums[i - 1] (1 <= i <= N),每个物品只有一个，请问你有几种不同的方法能够恰好装满这个背包？
     *
     * 第一步要明确两点：【状态】和【选择】。
     *      对于背包问题，这个都是一样的，状态就是【背包的容量】和【可选择的物品】，选择就是【装进背包】和【不装进背包】。
     * 第二步要明确 dp 数组的定义：
     *      dp[i][j] = x 表示，若只在前 i 个物品中选择， 若当前背包的容量为 j，则最多有x中方法可以恰好装满背包。
     *      根据这个定义，显然 dp[0][..] = 0,因为没有物品的话，根本没办法装背包； dp[..][0] = 1, 因为如果背包的最大载重为 0，【什么都不装】就是唯一的一种装法。
     *      我们所求的答案就是 dp[N][sum]， 即使用所有 N个物品，有几种方法可以装满容量为 sum 的背包。
     * 第三步，根据【选择】，思考状态转移的逻辑。
     *      如果不把 nums[i] 算入子集，或者说你不把这第 i 个物品装入背包，那么恰好装满背包的方法就取决于上一个状态 dp[i - 1][j],继承之前的结果。
     *      如果把 nums[i] 算入子集，或者说你把这第 i 个物品装入背包，那么只要看前 i-1 个物品有几种方法可以装满 j - nums[i - 1] 的重量就行了，所以取决于 dp[i-1][j-nums[i-1]]
     *       ps: 注意我们说的 i 是从1开始的，而数组 nums 的索引是从 0 开始算的，所以 nums[i - 1]代表的是第 i 个物品的重量，j - nums[i - 1]就是背包装入物品 i之后还剩下的重量。
     * 由于 dp[i][j] 为装满背包的总方法数，所以应该以上两种选择的结果求和，得到状态转移方程
     * dp[i][j] = dp[i-1][j] + dp[i-1][j-nums[i-1]]
     *
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
     *
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
     *
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

    // -------组合总和 start >>--------

    public List<List<Integer>> permute1(int[] nums) {
        return null;
    }

    // -------组合总和 << end --------

}
