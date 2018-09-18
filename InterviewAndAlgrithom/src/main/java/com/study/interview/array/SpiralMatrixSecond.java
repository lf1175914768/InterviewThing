package com.study.interview.array;

/**
 * Given a positive integer n, generate a square matrix filled with elements from 1 to n2 in spiral order.
 * 
 * @author Liufeng
 * @createData Created on: 2018年9月17日 下午1:35:15
 */
public class SpiralMatrixSecond {
	
	public int [][] generateMatrix(int n) {
		if(n <= 0) return null;
		int[][] result = new int[n][n];
		int p = n, row, col, val = 1;
		for(int i = 0; i < n / 2; i++, p -= 2) {
			for(col = i; col < i + p; col++) {
				result[i][col] = val++;
			}
			for(row = i + 1; row < i + p; row++) {
				result[row][i + p - 1] = val++;
			}
			for(col -= 2; col >= i; col--) {
				result[i + p - 1][col] = val++;
			}
			for(row -= 2; row > i; row--) {
				result[row][i] = val++;
			}
		}
		if(n % 2 == 1) result[n / 2][n / 2] = val;
		return result;
	}

}
