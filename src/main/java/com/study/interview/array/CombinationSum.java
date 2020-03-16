package com.study.interview.array;

import java.util.ArrayList;
import java.util.List;

/**
 * Find all possible combinations of k numbers that add up to a number n, 
 * given that only numbers from 1 to 9 can be used and each combination should be a unique set of numbers.
 * <p>Note:</p>
	All numbers will be positive integers.
	The solution set must not contain duplicate combinations
 * @author LiuFeng
 *
 */
public class CombinationSum {
	
	public List<List<Integer>> combinationSum3(int k, int n) {
		List<List<Integer>> result = new ArrayList<>();
		combination(result, new ArrayList<Integer>(), k, 1, n);
		return result;
	}

	private void combination(List<List<Integer>> result, ArrayList<Integer> comb,
			int count, int start, int number) {
		if(number == 0 && comb.size() == count) {
			List<Integer> li = new ArrayList<Integer>(comb);
			result.add(li);
			return;
		}
		for(int i = start; i <= 9; i++) {
			comb.add(i);
			combination(result, comb, count, i + 1, number - i);
			comb.remove(comb.size() - 1);
		}
	}

}
