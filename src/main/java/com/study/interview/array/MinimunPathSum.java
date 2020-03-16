package com.study.interview.array;

/**
 * Given a m x n grid filled with non-negative numbers, 
 * find a path from top left to bottom right which minimizes the sum of all numbers along its path.
 * 
 * Note: You can only move either down or right at any point in time.
 * 
 * @author Liufeng
 * @createData Created on: 2018年9月17日 下午2:17:55
 */
public class MinimunPathSum {
	
	public int minPathSum(int [][] grid) {
		if(grid == null || grid.length == 0 || grid[0].length == 0) return 0;
		int row = grid.length, col = grid[0].length;
		int[][] result = new int[row][col]; 
		result[0][0] = grid[0][0];
		for(int i = 1; i < row; i++) {
			result[i][0] = grid[i][0] + result[i - 1][0];
		}
		for(int i = 1; i < col; i++) {
			result[0][i] = grid[0][i] + result[0][i - 1];
		}
		for(int i = 1; i < row; i++) {
			for(int j = 1; j < col; j++) {
				result[i][j] = Math.min(result[i - 1][j], result[i][j - 1]) + grid[i][j];
			}
		}
		return result[row - 1][col - 1];
	}

}
