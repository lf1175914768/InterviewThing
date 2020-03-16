package com.study.interview.dynamicprogramming;

/**
 * 最长递增子数组的求法
 * @author LiuFeng
 *
 */
public class MaxIncrementalSubArray {
	
	//---------------------------------------------------
	//  一般的解法，时间复杂度是 n的平方
	//---------------------------------------------------
	
	public int[] list_1(int[] arr) {
		if(arr == null || arr.length == 0) {
			return null;
		}
		int[] dp = getdp_1(arr);
		return generateLIS(arr, dp);
	}
	
	public int[] generateLIS(int[] arr, int[] dp) {
		int index = 0;
		int len = 0;
		// 找到最大的数值对应的下标值
		for(int i = 1; i < dp.length; i++) {
			if(dp[i] > len) {
				len = dp[i];
				index = i;
			}
		}
		
		int[] list = new int[len];
		list[--len] = arr[index];
		for(int i = index; i >= 0; i--) {
			if(arr[i] < arr[index] && dp[i] == dp[index] - 1) {
				list[--len] = arr[i];
				index = i;
			}
		}
		return list;
	}
	
	public int[] getdp_1(int[] arr) {
		//  dp[i] 表示在以arr[i]这个数为结尾的情况下， arr[0... i]中的最大的递增子序列长度
		int[] dp = new int[arr.length];
		for(int i = 0; i < arr.length; i++) {
			dp[i] = 1;
			for(int j = 0; j < i; j++) {
				if(arr[i] > arr[j]) {
					dp[i] = Math.max(dp[i], dp[j] + 1);
				}
			}
		}
		return dp;
	}
	
	//-----------------------------------------------------------
	//  利用二分查找进行的优化，
	//-----------------------------------------------------------
	
	public int[] list_2(int[] arr) {
		if(arr == null || arr.length == 0) {
			return null;
		}
		int[] dp = getdp_2(arr);
		return generateLIS(arr, dp);
	}

	//  ends[0...right] 为有效区， 对有效区上的位置，如果有ends【b】 == c，
	// 表示遍历到目前位置， 在所有长度为 b + 1 的递增序列中， 最小的结尾数是 c。
	public int[] getdp_2(int[] arr) {
		int[] dp = new int[arr.length];
		int[] ends = new int[arr.length];
		ends[0] = arr[0];
		dp[0] = 1;
		int l = 0, r = 0, m = 0, right = 0;
		for(int i = 1; i < arr.length; i++) {
			l = 0;
			r = right;
			while(l <= r) {
				m = (l + r) / 2;
				if(arr[i] > ends[m]) {
					l = m + 1;
				} else {
					r = m - 1;
				}
			}
			right = Math.max(right, l);
			ends[l] = arr[i];
			dp[i] = l + 1;
		}
		return dp;
	}
}
