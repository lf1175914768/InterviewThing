package com.study.interview.chain;

/**
 * 将单链表的每 k 个节点之间逆序  
 * 例如： 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> null , K = 3;
 * 调整后为   3 -> 2 -> 1 -> 6 -> 5 -> 4 -> 7 -> 8 -> null
 * @author LiuFeng
 *
 */
public class ReverseKNodes {
	
	public Node reverseKNode(Node head, int k) {
		if(k < 2) {
			return head;
		}
		Node cur = head;
		Node pre = null;
		Node start = null;
		Node next = null;
		int count = 1;
		while(cur != null) {
			next = cur.next;
			if(count == k) {
				start = pre == null ? head : pre.next;
				head = pre == null ? cur : head;
				resign2(pre, start, cur, next);
				pre = start;
				count = 0;
			}
			count++;
			cur = next;
		}
		return head;
	}

	private void resign2(Node left, Node start, Node end, Node right) {
		Node pre = start;
		Node cur = start.next; 
		Node next = null;
		while(cur != right) {
			next = cur.next;
			cur.next = pre;
			pre = cur; 
			cur = next;
		}
		if(left != null) {
			left.next = end;
		}
		start.next = right;
	}

}
