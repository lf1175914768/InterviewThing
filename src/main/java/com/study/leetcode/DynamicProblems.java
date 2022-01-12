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

    // -------组合总和 start >>--------

    public List<List<Integer>> permute1(int[] nums) {
        return null;
    }

    // -------组合总和 << end --------

}
