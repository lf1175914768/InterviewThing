package com.study.book.algorithmtechnologymanual;

/**
 * java中两个数是如何相加的
 * @author LiuFeng
 *
 */
public class SumTwoInteger {
	
	public void add(int[] n1, int[] n2, int[] sum) {
		int i = n1.length - 1, j = n2.length - 1;
		int carry = 0;
		int length = Math.max(i, j) + 1;
		while(i >= 0 && j >= 0) {
			int total = n1[i] + n2[j] + carry;
			sum[length--] = total % 10;
			carry = total > 9 ? 1 : 0;
			i--; j--;
		}
		while(i >= 0) {
			int total = n1[i] + carry;
			sum[length--] = total % 10;
			carry = total > 9 ? 1 : 0;
			i--;
		}
		while(j >= 0) {
			int total = n2[j] + carry;
			sum[length--] = total % 10;
			carry = total > 9 ? 1 : 0;
			j--;
		}
		sum[0] = carry;
	}

}
