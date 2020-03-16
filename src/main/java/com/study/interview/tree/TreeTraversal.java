package com.study.interview.tree;

import java.util.Stack;

/**
 * 分别用递归和非递归的方法前序、 中序、 后序遍历二叉树
 * @author LiuFeng
 *
 */
public class TreeTraversal {
	
	//------------------------------------------------
	//  传统的递归方法
	//------------------------------------------------
	
	public void preOrderRecur(Node head) {
		if(head == null) {
			return;
		}
		System.out.println(head.value + " ");
		preOrderRecur(head.left);
		preOrderRecur(head.right);
	}
	
	public void inOrderRecur(Node head) {
		if(head == null) {
			return;
		}
		inOrderRecur(head.left);
		System.out.println(head.value + " ");
		inOrderRecur(head.right);
	}
	
	public void posOrderRecur(Node head) {
		if(head == null) {
			return;
		}
		posOrderRecur(head.left);
		posOrderRecur(head.right);
		System.out.println(head.value + " ");
	}
	
	//------------------------------------------------
	//  非递归方法
	//------------------------------------------------
	
	/**
	 * 用栈来实现前序遍历
	 * @param head
	 */
	public void preOrderUnRecur(Node head) {
		System.out.println("Pre-order: ");
		if(head != null) {
			Stack<Node> stack = new Stack<Node>();
			stack.add(head);
			while(!stack.isEmpty()) {
				head = stack.pop();
				System.out.println(head.value + " ");
				if(head.right != null) {
					stack.push(head.right); 
				} 
				if(head.left != null) {
					stack.push(head.left);
				}
			}
		}
		System.out.println();
	}
	
	/**
	 * 中序遍历
	 * @param head
	 */
	public void inOrderUnRecur(Node head) {
		System.out.println("In-order: ");
		if(head != null) {
			Stack<Node> stack = new Stack<Node>();
			while(!stack.isEmpty() || head != null) {
				if(head != null) {
					stack.push(head);
					head = head.left;
				} else {
					head = stack.pop();
					System.out.println(head.value + " ");
					head = head.right;
				}
			}
		}
	}
	
	/**
	 * 这里用的是两个栈进行后序遍历。
	 * @param head
	 */
	public void posOrderUnRecur_1(Node head) {
		System.out.println("pos-order: ");
		if(head != null) {
			Stack<Node> s1 = new Stack<Node>();
			Stack<Node> s2 = new Stack<Node>();
			s1.push(head);
			while(!s1.isEmpty()) {
				head = s1.pop();
				s2.push(head);
				if(head.left != null) {
					s1.push(head.left);
				}
				if(head.right != null) {
					s2.push(head.right);
				}
			}
			while(!s2.isEmpty()) {
				System.out.println(s2.pop().value + " ");
			}
		}
		System.out.println();
	}
	
	/**
	 * 这里用一个栈来实现后序遍历
	 * @param h
	 */
	public void posOrderUnRecur_2(Node h) {
		System.out.println("pos-order:　");
		if(h != null) {
			Stack<Node> stack = new Stack<Node>();
			stack.push(h);
			Node c = null;
			while(!stack.isEmpty()) {
				c = stack.peek();
				if(c.left != null && h != c.left && h != c.right) {
					stack.push(c.left);
				} else if(c.right != null && h != c.right) {
					stack.push(c.right);
				} else {
					System.out.println(stack.pop().value + " ");
					h = c;
				}
			}
		}
		System.out.println();
	}

}
