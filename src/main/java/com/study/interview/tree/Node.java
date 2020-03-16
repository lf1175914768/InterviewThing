package com.study.interview.tree;

public class Node {
	
	int value;
	Node left;
	Node right;
	
	public Node() {}
	public Node(int data) {
		this.value = data;
	}
	
	public Node(int data, Node left, Node right) {
		this.value = data;
		this.left = left;
		this.right = right;
	}
	
	@Override
	public String toString() {
		return String.valueOf(value);
	}

}
