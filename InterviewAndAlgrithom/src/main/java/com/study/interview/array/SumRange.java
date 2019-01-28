package com.study.interview.array;

import java.util.ArrayList;
import java.util.List;

/**
 * @author LiuFeng
 */
public class SumRange {
	
	public List<String> summaryRanges(int[] nums) {
		List<String> result = new ArrayList<>();
		if(nums.length <= 0) return result;
		StringBuilder sb = new StringBuilder(String.valueOf(nums[0]));
		int prev = nums[0], start = prev;
		for(int i = 1; i < nums.length; i++) {
			if(nums[i] - prev == 1) {
				prev = nums[i];
				continue;
			} 
			if(start != prev) {
				sb.append("->").append(prev);
			}
			result.add(sb.toString());
			sb = sb.delete(0, sb.length());
			prev = start = nums[i];
			sb.append(start);
		}
		if(start != prev) {
			sb.append("->").append(prev);
		}
		result.add(sb.toString());
		return result;
	}
	
	public List<String> summaryRanges_eayUnderstand(int[] nums) {
		List<String> result = new ArrayList<>();
		if(nums != null && nums.length > 0) {
			for(int i = 0; i < nums.length; i++) {
				int a = nums[i];
				while(i + 1 < nums.length && nums[i + 1] - nums[i] == 1) i++;
				if(a != nums[i]) {
					result.add(a + "->" + nums[i]);
				} else {
					result.add(a + "");
				}
			}
		}
		return result;
	}

}
