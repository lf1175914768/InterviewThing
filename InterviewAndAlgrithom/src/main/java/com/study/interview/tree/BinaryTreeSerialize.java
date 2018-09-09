package com.study.interview.tree;

import java.util.Stack;

/**
 * 这里用来实现二叉树的序列化和反序列化
 * @author Liufeng
 * Created on 2018年9月9日 下午10:02:34
 */
public class BinaryTreeSerialize {
	
	public String preOrderSerialize_1(Node root) {
		StringBuilder res = new StringBuilder();
		preOrder_1(root, res);
		return res.toString();
	}
	
	/**
	 * 这里使用递归来实现锝
	 */
	private void preOrder_1(Node root, StringBuilder res) {
		if(root == null) {
			res.append("#!");
			return;
		}
		res.append(root.value + "!");
		preOrder_1(root.left, res);
		preOrder_1(root.right, res);
	}
	
	public String preOrderSerialize_2(Node root) {
		StringBuilder sb = new StringBuilder();
		Stack<Node> stack = new Stack<Node>();
		Node node = root;
		while(!stack.isEmpty() || node != null) {
			if(node != null) {
				sb.append(node.value + "!");
				stack.push(node);
				node = node.left;
			} else {
				sb.append("#!");
				node = stack.pop();
				node = node.right;
			}
		}
		sb.append("#!");
		return sb.toString();
	}
	
	
	public String preOrderSerialize_3(Node root) {
		StringBuilder sb = new StringBuilder();
		Node node = root;
		Stack<Node> stack = new Stack<Node>();
		stack.push(node);
		while(!stack.isEmpty()) {
			node = stack.pop();
			if(node == null) {
				sb.append("#!");
			} else {
				sb.append(node.value + "!");
				stack.push(node.right);
				stack.push(node.left);
			}
		}
		return sb.toString();
	}
	
	public Node preOrderDeserialize_1(String str) {
		if(str == null || str.length() == 0) return null;
		String[] arr = str.split("!");
		return deserializeCore(arr);
	}
	
	private int index = 0;

	private Node deserializeCore(String[] arr) {
		if("#".equals(arr[index])) {
			index++;
			return null;
		} else {
			Node node = new Node(Integer.parseInt(arr[index]));
			index++;
			node.left = deserializeCore(arr);
			node.right = deserializeCore(arr);
			return node;
		}
	}
}
