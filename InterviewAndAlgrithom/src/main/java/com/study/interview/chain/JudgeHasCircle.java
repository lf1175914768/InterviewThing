package com.study.interview.chain;

/**
 * 判断一个链表里面是否有环
 * @author liufeng2
 *
 */
public class JudgeHasCircle {

	public int getCircleLength(Node root) {
		if(root == null) return 0;
		Node first = root, second = root;
		int visitCount = 0, meetCount = 0;
		int firstMeet = 0, secondMeet = 0;
		do {
			if(first == null || first.next == null ||
					second == null || second.next == null || second.next.next == null) return 0;
			first = first.next;
			second = second.next.next;
			visitCount++;
			if(first == second) {
				meetCount++;
				if(meetCount == 1) {
					firstMeet = visitCount;
				} else if(meetCount == 2) {
					secondMeet = visitCount;
				}
			}
		} while(meetCount < 2);
		return secondMeet - firstMeet;
	}
	
}
