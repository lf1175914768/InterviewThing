package com.study.interview.array;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * @author liufeng
 * @version 2020/5/2
 **/
public class TwoSum {

    public int[] twoSum(int[] nums, int target) {
        int[] result = {-1, -1};
        if(nums == null || nums.length == 0)
            return result;
        for(int i = 0; i < nums.length; i++) {
            int tt = target - nums[i], start = nums.length;;
            while(--start > i && nums[start] != tt);
            if(start > i) {
                result[0] = i;
                result[1] = start;
            }
        }
        return result;
    }
}
