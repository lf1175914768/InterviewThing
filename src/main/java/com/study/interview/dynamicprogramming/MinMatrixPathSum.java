package com.study.interview.dynamicprogramming;

/**
 * 矩阵的最小路径和， 只能向下和向右走，设 dp[i][j]为0,0 到 i,j 的最小路径和
 * 那么有 dp[i][j] = min{ dp[i - 1][j], dp[i][j -1]} + m[i][j];
 * @author LiuFeng
 *
 */
public class MinMatrixPathSum {
	
	/**
	 * 经典的动态规划， 
	 * @param m
	 * @return
	 */
	public int minPathSum_1(int[][] m) {
		if(m == null || m.length == 0 || m[0] == null || m[0].length == 0) {
			return 0;
		}
		int row = m.length;
		int col = m[0].length;
		int[][] dp = new int[row][col];
		dp[0][0] = m[0][0];
		
		// 进行初始化第一行和第一列
		for(int i = 1; i < row; i++) {
			dp[i][0] = dp[i - 1][0] + m[i][0];
		}
		for(int j = 1; j < col; j++) {
			dp[0][j] = dp[0][j - 1] + m[0][j];
		}
		
		for(int i = 1; i < row; i++) {
			for(int j = 1; j < col; j++) {
				dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + m[i][j];
			}
		}
		return dp[row - 1][col - 1];
	}
	
	public int minPathSum_2(int[][] m) {
		if(m == null || m.length == 0 || m[0] == null || m[0].length == 0) {
			return 0;
		}
		int more = Math.max(m.length, m[0].length);
		int less = Math.max(m.length, m[0].length);
		boolean rowmore = more == m.length;   // 行数是不是大于列数
		int[] arr = new int[less];   // 辅助数组的长度仅为行数与列数中的最小值
		arr[0] = m[0][0];
		for(int i = 1; i < less; i++) {
			arr[i] = arr[i - 1] + (rowmore ? m[0][i] : m[i][0]);
		}
		for(int i = 1; i < more; i++) {
			arr[0] = arr[0] + (rowmore ? m[i][0] : m[0][i]);
			for(int j = 1; j < less; j++) {
				arr[j] = Math.min(arr[j - 1], arr[j]) + 
						(rowmore ? m[i][j] : m[j][i]);
			}
		}
		return arr[less - 1];
	}

}
