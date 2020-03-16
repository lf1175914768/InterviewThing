package com.study.interview.array;

/**
 * Given an array of n positive integers and a positive integer s, 
 * find the minimal length of a contiguous subarray of which the sum ≥ s. 
 * If there isn't one, return 0 instead.
 * 
 * Example: 
 * 	Input: s = 7, nums = [2,3,1,2,4,3]
	Output: 2
	Explanation: the subarray [4,3] has the minimal length under the problem constraint.
 * @author liufeng
 *
 */
public class MinSubArray {
	
	// ----------------------------------
	// O(n) 时间复杂度
	// ----------------------------------
	
	public int minSubArrayLen(int s, int[] nums) {
		int left = 0, right = 0, len = nums.length, sum = 0, res = len + 1;
		while(right < len) {
			while(sum < s && right < len) {
				sum += nums[right++];
			}
			while(sum >= s) {
				res = Math.min(res, right - left);
				sum -= nums[left++];
			}
		}
		return res == len + 1 ? 0 : res;
	}
	
	// ----------------------------------
	// O(nlog(n)) 时间复杂度
	// ----------------------------------
	
	public int minSubArrayLen_2(int s, int[] nums) {
		int len = nums.length, res = len + 1;
		int[] sums = new int[res];
		for(int i = 1; i < res; i++) {
			sums[i] = sums[i - 1] + nums[i - 1];
		}
		for(int i = 0; i < len; i++) {
			int left = i + 1, right = len, t = sums[i] + s;
			while(left <= right) {
				int mid = left + ((right - left) >> 1);
				if(sums[mid] < t) 
					left = mid + 1;
				else 
					right = mid - 1;
			}
			if(left == len + 1) break;
			res = Math.min(res, left - i);
		}
		return res == len + 1 ? 0 : res;
	}

}
