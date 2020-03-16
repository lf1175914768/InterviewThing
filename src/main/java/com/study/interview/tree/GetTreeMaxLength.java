package com.study.interview.tree;

import java.util.HashMap;

/**
 * 在二叉树中找到累加和为指定值的最长路径长度。
 * @author LiuFeng
 *
 */
public class GetTreeMaxLength {
	
	public int getMaxLength(Node head, int sum) {
		HashMap<Integer, Integer> sumMap = new HashMap<Integer, Integer>();
		sumMap.put(0, 0);      // 重要
		return preOrder(head, sum, 0, 1, 0, sumMap);
	}

	private int preOrder(Node head, int sum, int preSum, int level, 
			int maxLen, HashMap<Integer, Integer> sumMap) {
		if(head == null) {
			return maxLen;
		}
		int curSum = preSum + head.value;
		if(!sumMap.containsKey(curSum)) {
			sumMap.put(curSum, level);
		}
		if(sumMap.containsKey(curSum - sum)) {
			maxLen = Math.max(level - sumMap.get(curSum - sum), maxLen);
		}
		maxLen = preOrder(head.left, sum, curSum, level + 1, maxLen, sumMap);
		maxLen = preOrder(head.right, sum, curSum, level + 1, maxLen, sumMap);
		if(level == sumMap.get(curSum)) {
			sumMap.remove(curSum);
		}
		return maxLen;
	}

}
