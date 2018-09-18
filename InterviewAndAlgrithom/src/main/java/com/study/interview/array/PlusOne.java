package com.study.interview.array;

/**
 * Given a non-empty array of digits representing a non-negative integer, plus one to the integer.
 * The digits are stored such that the most significant digit is at the head of the list, 
 * and each element in the array contain a single digit.
 * You may assume the integer does not contain any leading zero, except the number 0 itself.
 * 
 * @author Liufeng
 * @createData Created on: 2018年9月17日 下午2:37:21
 */
public class PlusOne {
	
	public int [] plusOne(int[] digits) {
		int length = digits.length;
		for(int i = length - 1; i >= 0; i--) {
			if(digits[i] < 9) {
				digits[i]++;
				return digits;
			} 
			digits[i] = 0;
		}
		int[] res = new int[length + 1];
		res[0] = 1;
		return res;
	}

}
