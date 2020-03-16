package com.study.interview.dynamicprogramming;

/**
 * 计算最大子数组或最大子矩阵的和。
 * @author liufeng2
 *
 */
public class MaxSumOfSubArray {
	
	/**
	 * 假设dp[i] 为以 array[i]为最后一个数字的规划数组，那么对于 dp[i]的求解， 
	 * 假如dp[i - 1] > 0的话， 那么dp[i] 的值就等于 dp[i - 1] + array[i]
	 * 否则 dp[i] 的值就等于 array[i]， 即以array[i]开始的子数组。
	 */
	public int maxOneDimensionSum(int[] array) {
		int length, result = Integer.MIN_VALUE;
		if(array == null || (length = array.length) == 0) return 0;
		int[] dp = new int[length];
		dp[0] = array[0];
		for(int i = 1; i < length; i++) {
			if(dp[i - 1] > 0) {
				dp[i] = dp[i - 1] + array[i];
			} else {
				dp[i] = array[i];
			}
			if(dp[i] > result) {
				result = dp[i];
			}
		}
		return result;
	}
	
	/**
	 * 对上面的简化。
	 */
	public int maxOneDimensionSum_2(int[] array) {
		int length, result = Integer.MIN_VALUE;
		if(array == null || (length = array.length) == 0) return 0;
		int dp = array[0];
		for(int i = 1; i < length; i++) {
			if(dp > 0) {
				dp += array[i];
			} else {
				dp = array[i];
			} 
			if(dp > result) result = dp;
		}
		return result;
	}
	
	

}
