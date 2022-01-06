package com.study.leetcode;

import java.util.Arrays;
import java.util.List;

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
     * @param nums nums
     * @return max length of longest increasing subsequence
     */
    public int lengthOfLIS(int[] nums) {
        int[] dp = new int[nums.length];
        // base case: dp数组全部初始化为1
        Arrays.fill(dp, 1);
        int res = 0;
        for (int i = 1; i < nums.length; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j])
                    dp[i] = Math.max(dp[i], 1 + dp[j]);
            }
            res = Math.max(res, dp[i]);
        }
        return res;
    }

    // -------最长递增子序列 << end --------

    // -------组合总和 start >>--------

    public List<List<Integer>> permute1(int[] nums) {
        return null;
    }

    // -------组合总和 << end --------

}
