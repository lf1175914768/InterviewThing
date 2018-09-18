package com.study.interview.array;

/**
 * Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.
 * Find the minimum element.
 * The array may contain duplicates.
 * 
 * @author Liufeng
 * @createData Created on: 2018年9月18日 下午2:39:36
 */
public class MinimunInRotatedSortedArray {
	
	// 有点小问题， 先这样， 回头想想解决方案。
	public int findMin(int[] nums) {
		if(nums == null || nums.length == 0) return 0;
		int left = 0, right = nums.length - 1, mid, result = nums[0];
		while(left < right) {
			mid = left + ((right - left) >> 1);
			if(nums[left] < nums[mid]) {
				result = Math.min(result, nums[left]);
				left = mid + 1; 
			} else if(nums[left] > nums[mid]) {
				result = Math.min(result, nums[right]);
				right = mid;
			} else ++left;
		}
		return result;
	}

}
