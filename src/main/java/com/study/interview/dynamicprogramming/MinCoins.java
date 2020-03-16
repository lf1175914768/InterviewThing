package com.study.interview.dynamicprogramming;

/**
 * 换钱的最少货币数  dp[i][j] = min{ dp[i - 1][j], dp[i][j - arr[i]] + 1 }
 * @author LiuFeng
 *
 */
public class MinCoins {
	
	/**
	 * 用一个二维数组进行计算
	 * @param arr
	 * @param aim
	 * @return
	 */
	public int minCoins_1(int[] arr, int aim) {
		if(arr == null || arr.length == 0 || aim < 0) {
			return -1;
		}
		int n = arr.length;
		int max = Integer.MAX_VALUE;
		int[][] dp = new int[n][aim + 1];
		for(int j = 1; j <= aim; j++) {
			dp[0][j] = max;
			if(j - arr[0] >= 0 && dp[0][j - arr[0]] != max) {
				dp[0][j] = dp[0][j - arr[0]] + 1;
			}
		}
		int left = 0;
		for(int i = 1; i < n; i++) {
			for(int j = 1; j <= aim; j++) {
				left = max;
				if(j - arr[i] >= 0 && dp[i][j - arr[i]] != max) {
					left = dp[i][j - arr[i]] + 1;
				}
				dp[i][j] = Math.min(left, dp[i - 1][j]);
			}
		}
		return dp[n - 1][aim] != max ? dp[n - 1][aim] : -1;
	}
	
	/**
	 * 用一个一维数组进行计算， 与上面的一样
	 * @param arr
	 * @param aim
	 * @return
	 */
	public int minCoins_2(int[] arr, int aim) {
		if(arr == null || arr.length == 0 || aim < 0) {
			return -1;
		}
		int n = arr.length;
		int max = Integer.MAX_VALUE;
		int[] dp = new int[aim + 1];
		for(int j = 1; j <= aim; j++) {
			dp[j] = max;
			if(j - arr[0] >= 0 && dp[j - arr[0]] != max) {
				dp[j] = dp[j - arr[0]] + 1;
			}
		}
		int left = 0;
		for(int i = 1; i < n; i++) {
			for(int j = 1; j <= aim; j++) {
				left = max;
				if(j - arr[i] >= 0 && dp[j - arr[i]] != max) {
					left = dp[j - arr[i]] + 1;
				}
				dp[j] = Math.min(left, dp[j]);  // 参考矩阵的路径最小和问题， 知道这里应该用的是dp[j], 而不是dp[j - 1]
			}
		}
		return dp[aim] != max ? dp[aim] : -1;
	}
	
	
	//--------------------------------------------
	// arr[i] 只有一张，  不能重复使用
	//--------------------------------------------
	
	public int minCoins_3(int [] arr, int aim) {
		if(arr == null || arr.length == 0 || aim < 0) {
			return -1;
		}
		int n = arr.length;
		int max = Integer.MAX_VALUE;
		int[][] dp = new int[n][aim + 1];
		for(int j = 1; j <= aim; j++) {
			dp[0][j] = max; 
		}
		if(arr[0] <= aim) {
			dp[0][arr[0]] = 1;
		}
		int leftup = 0;  // 左上角的某个位置的值
		for(int i = 1; i < n; i++) {
			for(int j = 1; j <= aim; j++) {
				leftup = max;
				if(j - arr[i] >= 0 && dp[i - 1][j - arr[i]] != max) {
					leftup = dp[i - 1][j - arr[i]] + 1;
				}
				dp[i][j] = Math.min(leftup, dp[i - 1][j]);
			}
		}
		return dp[n - 1][aim] != max ? dp[n - 1][aim] : -1;
	}
	
	public int minCoins_4(int [] arr, int aim) {
		if(arr == null || arr.length == 0 || aim < 0) {
			return -1;
		}
		int n = arr.length;
		int max = Integer.MAX_VALUE;
		int[] dp = new int[aim + 1];
		for(int j = 1; j <= aim; j++) {
			dp[j] = max; 
		}
		if(arr[0] <= aim) {
			dp[arr[0]] = 1;
		}
		int leftup = 0;  // 左上角的某个位置的值
		for(int i = 1; i < n; i++) {
			for(int j = aim; j > 0; j--) {
				leftup = max;
				if(j - arr[i] >= 0 && dp[j - arr[i]] != max) {
					leftup = dp[j - arr[i]] + 1;
				}
				dp[j] = Math.min(leftup, dp[j]);
			}
		}
		return dp[aim] != max ? dp[aim] : -1;
	}

}
 