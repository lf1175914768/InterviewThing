package com.study.book.algorithmtechnologymanual;

import org.junit.Test;

public class SumTwoIntegerTest {
	
	@Test
	public void testSum() {
		SumTwoInteger test = new SumTwoInteger();
//		int[] n1 = {1,2,3,4}, n2 = {3,4,5};
		int[] n1 = {9,9,9,9,9,9}, n2 = {9,9,9,9,9,9,9};
		int[] sum = new int[Math.max(n1.length, n2.length) + 1];
		test.add(n1, n2, sum);
		for(int num : sum) {
			System.out.println(num);
		}
	}
	
	@Test
	public void testMulti() {
		MultiTwoInteger test = new MultiTwoInteger();
		int[] n1 = {1,0, 2,4}, n2 = {3,2};
		int[] result = new int[6];
		test.mult(n1, n2, result);
		for(int num : result) {
			System.out.println(num);
		}
	}

}
