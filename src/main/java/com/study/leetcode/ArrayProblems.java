package com.study.leetcode;

import java.util.HashMap;
import java.util.Map;

/**
 * <p>description: 数组相关的问题  </p>
 * <p>className:  ArrayProblems </p>
 * <p>create time:  2022/4/6 11:28 </p>
 *
 * @author feng.liu
 * @since 1.0
 **/
public class ArrayProblems {

    // -------和为K的子数组 start >>--------

    /**
     * 给你一个整数数组 nums 和一个整数 k ，请你统计并返回 该数组中和为 k 的子数组的个数 。
     *
     * 首先使用前缀和技巧 解答。
     *
     * 对应 leetcode 中第 560 题。
     */
    public int subarraySum(int[] nums, int k) {
        int n = nums.length; 
        int[] preSum = new int[n + 1];
        for (int i = 1; i < preSum.length; i++) {
            preSum[i] = preSum[i - 1] + nums[i - 1];
        }

        int res = 0;
        for (int i = 1; i <= n; i++) {
            for (int j = 0; j < i; j++) {
                // 子数组 nums[j..i - 1] 的元素和
                if (preSum[i] - preSum[j] == k)
                    res++;
            }
        }
        return res;
    }

    /**
     * 上面的方法 优化版
     *
     * 优化的思路是： 我直接记录下有几个 preSum[j] = preSum[i] - k ，直接更新结果，避免内层的循环
     *
     * 其中 preSum 保存的是 前缀和 以及 对应出现的次数，
     * 比如对于 nums   [3, 5, 2, -2, 4, 1]
     * 对应的 前缀和 [0, 3, 8, 10, 8, 12, 13], 因为第一个数字一定是0，所以有 base case：preSum.put(0, 1)
     */
    public int subarraySum_v2(int[] nums, int k) {
        int n = nums.length;
        // map: 前缀和 -> 该前缀和出现的次数
        Map<Integer, Integer> preSum = new HashMap<>();
        // base case
        preSum.put(0, 1);
        int res = 0, sum = 0;
        for (int i = 0; i < n; i++) {
            sum += nums[i];
            // 这是我们想找的前缀和 nums[0..j]
            int sum0_j = sum - k;
            if (preSum.containsKey(sum0_j)) {
                res += preSum.get(sum0_j);
            }
            // 把前缀和 nums[0..i] 加入并记录出现次数
            preSum.put(sum, preSum.getOrDefault(sum, 0) + 1);
        }
        return res;
    }

    // -------和为K的子数组 << end --------
}
