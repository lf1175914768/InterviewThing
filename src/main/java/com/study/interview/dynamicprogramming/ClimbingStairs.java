package com.study.interview.dynamicprogramming;

/**
 * 有 N 阶楼梯，每次可以上一阶或者两阶，求有多少种上楼梯的方法 (easy)
 * @date: 2020/4/1 22:25
 * @author: Liufeng
 **/
public class ClimbingStairs {

    /**
     * 定义一个数组dp[i] 表示走到第i个楼梯的方法数目。
     * 第i个楼梯可以从第 i - 1 和 i - 2 个楼梯在走一步到达，走到第i个楼梯的方法数为
     * 走到第 i - 1 个和第 i - 2 个楼梯的方法之和。
     * dp[i] = dp[i - 1] + dp[i - 2]
     */
    public int climbStairs(int n) {
        if(n <= 2) {
            return n;
        }
        int pred = 1, last = 2, tmp;
        for(int i = 3; i <= n; i++) {
            tmp = pred + last;
            pred = last;
            last = tmp;
        }
        return last;
    }
}
