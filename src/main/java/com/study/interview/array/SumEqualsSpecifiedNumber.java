package com.study.interview.array;

import java.util.HashMap;

/**
 * 未排序数组中， 累加和为给定值的最长子数组的系列问题
 * @author LiuFeng
 *
 */
public class SumEqualsSpecifiedNumber {
	
	public int maxLength(int[] arr, int k) {
		if(arr == null || arr.length == 0) {
			return 0;
		}
		HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
		map.put(0, -1);      // 重要， 表示当位置在 -1 的时候， 前 n 项和为0 ， 是为了避免 漏掉arr[0] 这个元素
		
		int sum = 0;
		int len = 0;
		for(int i = 0; i < arr.length; i++) {
			sum += arr[i];
			if(map.containsKey(sum - k)) {
				len = Math.max(i - map.get(sum - k), len);
			}
			if(!map.containsKey(sum)) {
				map.put(sum, i);
			}
		}
		return len;
	}

}
