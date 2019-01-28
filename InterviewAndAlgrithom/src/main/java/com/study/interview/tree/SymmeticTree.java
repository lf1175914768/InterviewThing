package com.study.interview.tree;

import java.util.LinkedList;
import java.util.Queue;
import java.util.Stack;

public class SymmeticTree {
	
	//------------------------------------
	// solve recursively
	//------------------------------------
	
	public boolean isSymmetic(Node root) {
		if(root == null) return true;
		return isSymmetic(root.left, root.right);
	}

	private boolean isSymmetic(Node left, Node right) {
		if(left == null && right == null) return true;
		if(left == null && right != null
				|| left != null && right == null
				|| left.value != right.value) 
			return false;
		return isSymmetic(left.left, right.right) &&
				isSymmetic(right.left, left.right);
	}
	
	//------------------------------------
	// solve iteratively
	//------------------------------------

	public boolean isSymmetic_2(Node root) {
		if(root == null) return true;
		Queue<Node> q1 = new LinkedList<Node>();
		Queue<Node> q2 = new LinkedList<Node>();
		q1.offer(root.left);
		q2.offer(root.right);
		while(!q1.isEmpty() && !q2.isEmpty()) {
			Node node1 = q1.poll();
			Node node2 = q2.poll();
			if(node1 == null && node2 == null) 
				continue;
			if(node1 == null && node2 != null 
					|| node1 != null && node2 == null
					|| node1.value != node2.value) 
				return false;
			q1.offer(node1.left);
			q1.offer(node1.right);
			q2.offer(node2.left);
			q2.offer(node2.right);
		}
		if(!q1.isEmpty() || !q2.isEmpty()) return false;
		return true;
	}
	
	public boolean isSymmetic_3(Node root) {
		if(root == null) return true;
		Stack<Node> stack = new Stack<Node>();
		Node left, right;
		/**
		 * 下面注释的部分可以直接用 push 两个root 来解决
		 */
		/*
		if(root.left != null) {
			if(root.right == null) return false;
			stack.push(root.left);
			stack.push(root.right);
		} else if(root.right != null) {
			return false; 
		}  
		*/
		stack.push(root);
		stack.push(root);
		while(!stack.isEmpty()) {
			right = stack.pop();
			left = stack.pop();
			if(right.value != left.value) 
				return false;
			if(left.left != null) {
				if(right.right == null) return false;
				stack.push(left.left);
				stack.push(right.right);
			} else if(right.right != null) 
				return false;
			if(left.right != null) {
				if(right.left == null) return false;
				stack.push(left.right);
				stack.push(right.left);
			} else if(right.left != null) 
				return false;
		}
		return true;
	}
	
}
