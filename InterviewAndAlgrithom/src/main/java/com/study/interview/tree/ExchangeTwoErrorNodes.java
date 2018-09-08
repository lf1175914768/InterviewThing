package com.study.interview.tree;

import java.util.Stack;

/**
 * 调整搜索二叉树中两个错误的节点
 * @author LiuFeng
 *
 */
public class ExchangeTwoErrorNodes {
	
	public Node[] getTwoErrNodes(Node head) {
		Node[] errors = new Node[2];
		if(head == null) {
			return errors;
		}
		Stack<Node> stack = new Stack<Node>();
		Node pre = null;
		while(!stack.isEmpty() || head != null) {
			if(head != null) {
				stack.push(head);
				head = head.left;
			} else {
				head = stack.pop();
				if(pre != null && pre.value > head.value) {
					errors[0] = errors[0] == null ? pre : errors[0];
					errors[1] = head;
				}
				pre = head;
				head = head.right;
			}
		}
		return errors;
	}

}
