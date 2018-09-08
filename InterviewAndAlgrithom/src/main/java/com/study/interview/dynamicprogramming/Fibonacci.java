package com.study.interview.dynamicprogramming;

/**
 * 斐波那契数列的经典求解
 * @author LiuFeng
 *
 */
public class Fibonacci {
	
	/**
	 * O(2 的n次方)
	 * @param n
	 * @return
	 */
	public int f1(int n) {
		if(n < 1) return 0;
		if(n == 1 || n == 2) return 1;
		return f1(n - 1) + f1(n -2);
	}
	
	/**
	 * 顺序求解   O(N)
	 */
	public int f2(int n) {
		if(n < 1) return 0;
		if(n == 1 || n == 2) return 1;
		int res = 1;
		int pre = 1;
		int tmp = 0;
		for(int i = 3; i <= n; i++) {
			tmp = res;
			res = res + pre;
			pre = tmp;
		}
		return res;
	}
	
	/**
	 * 通过整理归纳， 最后变成矩阵相乘的实现  (不好理解， 不做要求)
	 * @param n
	 * @return
	 */
	public int f3(int n) {
		if(n < 1) {
			return 0;
		}
		if(n == 1 || n == 2) {
			return 1;
		}
		int[][] base = { {1, 1} , {1, 0} };
		int[][] res = matrixPower(base, n - 2);
		return res[0][0] + res[1][0];
	}

	private int[][] matrixPower(int[][] m, int p) {
		int[][] res = new int[m.length][m[0].length];
		// 先把res设为单位矩阵， 相当于整数中的1
		for(int i = 0; i < res.length; i++) {
			res[i][i] = 1;
		}
		int[][] tmp = m;
		for(; p != 0; p >>= 1) {
			if((p & 1) != 0) {
				res = muliMatrix(res, tmp);
			}
			tmp = muliMatrix(tmp ,tmp);
		}
		return res;
	}

	private int[][] muliMatrix(int[][] m1, int[][] m2) {
		int[][] res = new int[m1.length][m2[0].length];
		for(int i = 0; i < m2[0].length; i++) {
			for(int j = 0; j < m1.length; j++) {
				for(int k = 0; k < m2.length; k++) {
					res[i][j] += m1[i][k] * m2[k][j];
				}
			}
		}
		return res;
	}

}
