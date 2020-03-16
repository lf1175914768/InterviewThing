package com.study.book.algorithmtechnologymanual;

/**
 * java的乘法运算
 * @author LiuFeng
 *
 */
public class MultiTwoInteger {
	
	public void mult(int[] n1, int[] n2, int[] result) {
		int pos = result.length - 1;
		
		// 清除所有的值
		for(int i = 0; i < result.length; i++) {
			result[i] = 0;
		}
		for(int m = n1.length - 1; m >= 0; m--) {
			int off = n1.length - 1 - m;
			for(int n = n2.length - 1; n >= 0; n--, off++) {
				int prod = n1[m] * n2[n];
				
				// 计算部分和， 并且加上进位
				result[pos - off] += prod % 10;
				result[pos - off - 1] += prod / 10 + result[pos - off] / 10;
				result[pos - off] %= 10;
			}
		}
	}

}
