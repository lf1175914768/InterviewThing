package com.study.interview.array;

import java.util.Arrays;

/**
 * 对于一个数组，找出其中的三个数，使其和最接近于给定的target
 * @author Liufeng
 * @createData Created on: 2018年9月13日 下午5:01:54
 */
public class ThreeSumCloset {

	public int threeSumCloset(int[] nums, int target) {
		if(nums == null || nums.length < 3) return -1;//表示情况不符合要求。
		int result = nums[0] + nums[1] + nums[2], temp, length = nums.length - 1,
				sum, diff = Integer.MAX_VALUE, a;
		Arrays.sort(nums);
		for(int i = 0; i < nums.length - 2; i++) {
			int left = i + 1, right = length;
			while(left < right) {
				sum = nums[i] + nums[left] + nums[right];
				if(diff > (a = Math.abs((temp = target - sum)))) {
					result = sum;
                    diff = a;
				} 
				if(temp == 0) {
					return sum;
				} else if(temp < 0) {
					--right;
				} else {
					++left;
				}
			}
		}
		return result;
	}
	
}
 