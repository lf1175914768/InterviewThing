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

    // -------爱吃香蕉的珂珂 start >>--------

    /**
     * 珂珂喜欢吃香蕉。这里有 N 堆香蕉，第 i 堆中有 piles[i] 根香蕉。警卫已经离开了，将在 H 小时后回来。
     * 珂珂可以决定她吃香蕉的速度 K （单位：根/小时）。每个小时，她将会选择一堆香蕉，从中吃掉 K 根。如果这堆香蕉少于 K 根，她将吃掉这堆的所有香蕉，
     * 然后这一小时内不会再吃更多的香蕉。 珂珂喜欢慢慢吃，但仍然想在警卫回来前吃掉所有的香蕉。
     * 返回她可以在 H 小时内吃掉所有香蕉的最小速度 K（K 为整数）。
     *
     * 根据题意可以知道：
     * 珂珂吃香蕉的速度越小，耗时越多。反之，速度越大，耗时越少，这是题目的单调性。
     * 我们要找的是速度，因为题目限制了珂珂一个小时之内只能选择一堆香蕉吃，因此速度最大值就是这几堆香蕉中，数量最多的那一堆。速度的最小值是 1，
     * 其实还可以在分析一下下界是多少，由于二分搜索的时间复杂度很低，严格的分析不是很有必要。
     * 还是因为珂珂一个小时之内只能选择一堆香蕉吃，因此：每堆香蕉吃完的耗时 = 这堆香蕉的数量 / 珂珂一小时吃香蕉的数量。根据题意，这里的 / 在不能
     * 整除的时候，需要 上取整。
     * 注意：
     * 当 【二分查找】算法猜测的速度恰好使得珂珂在规定的时间内吃完所有的香蕉的时候，还应该去尝试更小的速度是不是还可以保证在规定的时间内吃完香蕉。
     * 这是因为题目问的是 【最小速度】。
     *
     * 对应 leetcode 中第 875 题。
     */
    public int minEatingSpeed(int[] piles, int h) {
        int maxVal = 1;
        for (int pile : piles) {
            maxVal = Math.max(maxVal, pile);
        }
        // 速度最小的时候，耗时最长
        int left = 1;
        // 速度最大的时候，耗时最短
        int right = maxVal;
        while (left < right) {
            int mid = left + ((right - left) >>> 1);
            if (minEatingSpeedFunction(piles, mid) > h) {
                // 耗时太多，说明速度太慢了，下一轮搜索区间是 [mid + 1, right)
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return left;
    }

    private int minEatingSpeedFunction(int[] piles, int speed) {
        int hours = 0;
        for (int i = 0; i < piles.length; i++) {
            hours += piles[i] / speed;
            if (piles[i] % speed > 0) {
                hours++;
            }
        }
        return hours;
    }

    // -------爱吃香蕉的珂珂 << end --------

    // -------组合总和 start >>--------

    public List<List<Integer>> permute1(int[] nums) {
        return null;
    }

    // -------组合总和 << end --------
}
