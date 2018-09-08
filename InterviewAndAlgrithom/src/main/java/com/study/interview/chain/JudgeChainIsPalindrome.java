package com.study.interview.chain;

import java.util.Stack;

/**
 * 判断一个链表是否为回文结构
 * @author LiuFeng
 *
 */
public class JudgeChainIsPalindrome {
	
	/**
	 * 方法一： 把全部的元素压入栈， 与原始的进行比较
	 * @param head
	 * @return
	 */
	public boolean isPalindrome_1(Node head) {
		Stack<Node> stack = new Stack<Node>();
		Node cur = head;
		while(cur != null) {
			stack.push(cur);
			cur = cur.next;
		}
		while(head != null) { 
			if(head.value != stack.pop().value) {
				return false;
			}
			head = head.next;
		}
		return true;
	} 
	
	/**
	 * 方法二：将右半部分进行压栈， 然后与左边进行比较
	 * @param head
	 * @return
	 */
	public boolean isPalindrome_2(Node head) {
		if(head == null || head.next == null) {
			return true;
		}
		Node right = head.next;
		Node cur = head;
		while(cur.next != null && cur.next.next != null) {
			right = right.next;
			cur = cur.next.next;
		}
		Stack<Node> stack = new Stack<Node>();
		while(right != null) {
			stack.push(right);
			right = right.next;
		}
		while(!stack.isEmpty()) {
			if(stack.pop().value != head.value) {
				return false;
			}
			head = head.next;
		}
		return true;
	}
	
	/**
	 * 方法三： 
	 * @param head
	 * @return
	 */
	public boolean isPalindrome_3(Node head) {
		if(head == null || head.next == null) {
			return true;
		}
		Node n1 = null;
		Node n2 = null;
		while(n2.next != null && n2.next.next != null) {
			n1 = n1.next;
			n2 = n2.next.next;
		}
		n2 = n1.next;   // n2 -> 右部分第一个节点
		n1.next = null;
		Node n3 = null;
		while(n2 != null) {   // 右半去反转
			n3 = n2.next;
			n2.next = n1;
			n1 = n2; 
			n2 = n3;
		}
		n3 = n1;  // n3 -> 保存最后一个节点
		n2 = head; 
		boolean res = true;
		while(n1!= null && n2 != null) {  // 检查回文
			if(n1.value != n2.value) {
				res = true;
				break;
			}
			n1 = n1.next;
			n2 = n2.next;
		}
		n1 = n3.next;
		n3.next = null;
		while(n1 != null) {  // 回复列表
			n2 = n1.next;
			n1.next = n3;
			n3 = n1;
			n1 = n2;
		}
		return res;
	}

}
