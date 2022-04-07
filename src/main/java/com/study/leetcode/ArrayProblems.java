package com.study.leetcode;

import java.util.*;

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

    // -------下一个更大的元素II start >>--------

    /**
     * 给定一个循环数组 nums （ nums[nums.length - 1] 的下一个元素是 nums[0] ），返回 nums 中每个元素的 下一个更大元素 。
     * 数字 x 的 下一个更大的元素 是按数组遍历顺序，这个数字之后的第一个比它更大的数，这意味着你应该循环地搜索它的下一个更大的数。如果不存在，则输出 -1 。
     *
     * 使用单调栈的方法进行解答
     *
     * 对应 leetcode 中第 503 题。
     */
    public int[] nextGreaterElements(int[] nums) {
        Stack<Integer> stack = new Stack<>();
        int[] res = new int[nums.length];
        int len = nums.length;
        for (int i = 2 * len - 1; i >= 0; i--) {
            while (!stack.isEmpty() && stack.peek() <= nums[i % len]) {
                stack.pop();
            }
            if (stack.isEmpty()) {
                res[i % len] = -1;
            } else {
                res[i % len] = stack.peek();
            }
            stack.push(nums[i % len]);
        }
        return res;
    }

    // -------下一个更大的元素II << end --------

    // -------滑动窗口最大值 start >>--------

    /**
     * 给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。
     * 返回 滑动窗口中的最大值 。
     *
     * 对应 leetcode 中第 239 题
     */
    public int[] maxSlidingWindow(int[] nums, int k) {
        // 双向队列，保存当前窗口最大值的数组位置，保证队列中数组位置的数值按照从大到小的顺序排列
        Deque<Integer> queue = new LinkedList<>();
        // 结果数组
        int[] res = new int[nums.length - k + 1];
        for (int i = 0; i < nums.length; i++) {
            while (!queue.isEmpty() && nums[queue.peekLast()] <= nums[i]) {
                queue.pollLast();
            }
            queue.offerLast(i);
            // 判断当前队列中队首的值是否有效，也就是是否在窗口中，不满足的话，需要移除
            if (queue.peek() <= i - k) {
                queue.pollFirst();
            }
            // 当窗口长度为 k 时，保存当前窗口最大值
            if (i + 1 >= k) {
                res[i - k + 1] = nums[queue.peekFirst()];
            }
        }
        return res;
    }

    // -------滑动窗口最大值 << end --------

    // -------组合总和 start >>--------

    public List<List<Integer>> permute1(int[] nums) {
        return null;
    }

    // -------组合总和 << end --------
}
