package com.study.interview.dynamicprogramming;

import java.util.List;

/**
 * @author Liufeng
 * Created on 2018年9月8日 下午10:35:49
 */
public class Triangle {
	
	public int minimumTotal(List<List<Integer>> triangle) {
        if(triangle == null || triangle.size() == 0) return 0;
        int[] result = new int[triangle.size()];
        int preCur = 0, candidate, res = Integer.MAX_VALUE;
        for(List<Integer> row : triangle) {
            for(int i = 0, length = row.size(); i < length; i++) {
            	if(i == 0) {
            		preCur = result[i];
            		result[i] += row.get(i);
            	}
            	else if(i == length - 1)  result[i] = row.get(i) + preCur;
            	else {
            		candidate = Math.min(preCur, result[i]);
            		preCur = result[i];
            		result[i] = row.get(i) + candidate;
            	}
            }
        }
        for(int i : result) {
        	if(i < res) res = i;
        }
        return res;
    }

}
