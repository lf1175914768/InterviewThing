package com.study.interview.dynamicprogramming;

/**
 * 抢劫一排住户，但是不能抢邻近的住户，求最大抢劫量。(easy)
 * @date: 2020/4/1 22:39
 * @author: Liufeng
 **/
public class HouseBobber {

    /**
     * 定义dp数组用来存储最大的抢劫量，其中dp[i]表示抢到第i个住户时的最大抢劫量。
     * 由于不能抢劫临近住户，如果抢劫了第 i - 1 个住户，那么就不能再抢劫第i个住户，
     * 所以   dp[i] = max{dp[i - 2] + nums[i], dp[i - 1]}
     */
    public int rob(int[] nums) {
        int pre2 = 0, pre1 = 0, cur = 0;
        for(int i = 0; i < nums.length; i++) {
            cur = Math.max(pre2 + nums[i], pre1);
            pre2 = pre1;
            pre1 = cur;
        }
        return cur;
    }
}
