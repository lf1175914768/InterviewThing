package com.study.interview.tree;

import java.util.ArrayList;
import java.util.List;

public class BinaryTreePaths {
	
	public List<String> binaryTreePaths(Node root) {
		List<String> result = new ArrayList<>();
		if(root != null) helper(root, "", result);
		return result;
	}

	private void helper(Node node, String out, List<String> result) {
		if(node.left == null && node.right == null) result.add(out + String.valueOf(node.value));
		if(node.left != null) helper(node.left, out + String.valueOf(node.value) + "->", result);
		if(node.right != null) helper(node.right, out + String.valueOf(node.value) + "->", result);
	}

}
 